#include <pcap.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <arpa/inet.h>
#include <netinet/if_ether.h>
#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <pthread.h>
#include "mongoose.h"


// --- IDS Configuration ---
#define PORT_SCAN_THRESHOLD 15
#define SCAN_TIME_WINDOW_SEC 5
#define MAX_TRACKED_IPS 100
#define MAX_UNIQUE_PORTS_PER_IP 256

// --- Global State ---
static pcap_t *g_pcap_handle = NULL;
volatile int g_is_sniffer_running = 0;
static struct mg_mgr g_mgr;
static struct mg_connection *g_ws_connection = NULL;

// --- Data Structures ---
typedef struct {
    in_addr_t ip_addr;
    time_t last_activity;
    unsigned short unique_ports[MAX_UNIQUE_PORTS_PER_IP];
    int port_count;
} IpScanState;

IpScanState ip_states[MAX_TRACKED_IPS];
int current_tracked_ips = 0;

// --- Function Prototypes ---
void packet_handler(u_char *user_data, const struct pcap_pkthdr *pkthdr, const u_char *packet);
int find_ip_state(in_addr_t ip);
void add_unique_port(int state_idx, unsigned short port);
void cleanup_old_states();
void log_and_broadcast_alert(const char* type, const char* src_ip, const char* dst_ip, const char* description);
void *sniffer_thread_func(void *dev_name_ptr);

// --- Alert Logging and WebSocket Broadcasting ---
void log_and_broadcast_alert(const char* type, const char* src_ip, const char* dst_ip, const char* description) {
    char json_buffer[512];
    time_t rawtime;
    struct tm *info;
    char timestamp[80];

    time(&rawtime);
    info = localtime(&rawtime);
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", info);

    snprintf(json_buffer, sizeof(json_buffer),
             "{\"timestamp\": \"%s\", \"type\": \"%s\", \"src_ip\": \"%s\", \"dest_ip\": \"%s\", \"details\": \"%s\"}",
             timestamp, type, src_ip, dst_ip, description);

    printf("ALERT: %s\n", json_buffer);

    if (g_ws_connection != NULL) {
        mg_ws_send(g_ws_connection, json_buffer, strlen(json_buffer), WEBSOCKET_OP_TEXT);
    }
}

// --- Packet Handling and IDS Logic ---
void packet_handler(u_char *user_data, const struct pcap_pkthdr *pkthdr, const u_char *packet) {
    const struct ip *ip_header = (struct ip*)(packet + sizeof(struct ether_header));
    char source_ip[INET_ADDRSTRLEN];
    char dest_ip[INET_ADDRSTRLEN];

    if (ntohs(((struct ether_header *)packet)->ether_type) != ETHERTYPE_IP) return;
    if (ip_header->ip_p != IPPROTO_TCP) return;

    inet_ntop(AF_INET, &(ip_header->ip_src), source_ip, INET_ADDRSTRLEN);
    inet_ntop(AF_INET, &(ip_header->ip_dst), dest_ip, INET_ADDRSTRLEN);
    unsigned short dest_port = ntohs(((struct tcphdr*)(packet + sizeof(struct ether_header) + (ip_header->ip_hl * 4)))->th_dport);

    int state_idx = find_ip_state(ip_header->ip_src.s_addr);
    if (state_idx == -1 && current_tracked_ips < MAX_TRACKED_IPS) {
        state_idx = current_tracked_ips++;
        ip_states[state_idx].ip_addr = ip_header->ip_src.s_addr;
        ip_states[state_idx].port_count = 0;
        memset(ip_states[state_idx].unique_ports, 0, sizeof(ip_states[state_idx].unique_ports));
    }

    if (state_idx != -1) {
        add_unique_port(state_idx, dest_port);
        ip_states[state_idx].last_activity = time(NULL);

        if (ip_states[state_idx].port_count >= PORT_SCAN_THRESHOLD) {
            char desc[200];
            sprintf(desc, "Hit %d unique ports in %d seconds.", ip_states[state_idx].port_count, SCAN_TIME_WINDOW_SEC);
            log_and_broadcast_alert("Port Scan", source_ip, dest_ip, desc);
            ip_states[state_idx].port_count = 0;
        }
    }
    cleanup_old_states();
}

// --- Sniffer Thread ---
void* sniffer_thread_func(void* dev_name_ptr) {
    char* dev = (char*)dev_name_ptr;
    char errbuf[PCAP_ERRBUF_SIZE];

    g_pcap_handle = pcap_open_live(dev, BUFSIZ, 1, 1000, errbuf);
    if (g_pcap_handle == NULL) {
        log_and_broadcast_alert("Error", "System", "N/A", errbuf);
        g_is_sniffer_running = 0;
        free(dev);
        return NULL;
    }

    log_and_broadcast_alert("Info", "System", "N/A", "Sniffer started.");
    pcap_loop(g_pcap_handle, -1, packet_handler, NULL);

    pcap_close(g_pcap_handle);
    g_pcap_handle = NULL;
    g_is_sniffer_running = 0;
    log_and_broadcast_alert("Info", "System", "N/A", "Sniffer stopped.");
    free(dev);
    return NULL;
}

// --- Embedded Frontend HTML ---
static const char *s_html_template =
    "<!DOCTYPE html>"
    "<html>"
    "<head>"
    "<title>IDS Dashboard</title>"
    "<style>"
    "body { font-family: 'Segoe UI', sans-serif; background-color: #2c2c2c; color: #f1f1f1; margin: 0; padding: 20px; }"
    "h1 { text-align: center; color: #00aaff; }"
    ".container { max-width: 1200px; margin: auto; background-color: #3a3a3a; padding: 20px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.4); }"
    ".controls, .actions { background-color: #444; padding: 15px; border-radius: 5px; margin-bottom: 20px; display: flex; align-items: center; gap: 15px; }"
    "label { font-weight: bold; }"
    "input[type='text'] { background-color: #222; color: #eee; border: 1px solid #555; padding: 8px; border-radius: 4px; }"
    "button { background-color: #007acc; color: white; border: none; padding: 8px 15px; border-radius: 4px; cursor: pointer; transition: background-color 0.3s; }"
    "button:hover { background-color: #005f99; }"
    "button.stop { background-color: #d9534f; }"
    "button.stop:hover { background-color: #c9302c; }"
    "table { width: 100%; border-collapse: collapse; margin-top: 20px; }"
    "th, td { padding: 12px; text-align: left; border-bottom: 1px solid #555; }"
    "th { background-color: #007acc; }"
    "tbody tr { cursor: pointer; }"
    "tbody tr:hover { background-color: #4a4a4a; }"
    "tbody tr.selected { background-color: #005f99; }"
    "#status { margin-top: 15px; background-color: #222; padding: 10px; border-radius: 4px; text-align: center; }"
    "</style>"
    "</head>"
    "<body>"
    "<div class='container'>"
    "<h1>IDS Real-Time Dashboard</h1>"
    "<div class='controls'>"
    "<label for='iface'>Interface:</label>"
    "<input type='text' id='iface' value='lo'>"
    "<button id='startBtn'>Start Sniffer</button>"
    "<button id='stopBtn' class='stop'>Stop Sniffer</button>"
    "</div>"
    "<div class='actions'>"
    "<button id='blockBtn'>Block Selected IP</button>"
    "<button id='unblockBtn'>Unblock Selected IP</button>"
    "</div>"
    "<table id='alertsTable'>"
    "<thead><tr><th>Timestamp</th><th>Type</th><th>Source IP</th><th>Dest IP</th><th>Details</th></tr></thead>"
    "<tbody></tbody>"
    "</table>"
    "<div id='status'>Status: Idle</div>"
    "</div>"
    "<script>"
    "const startBtn = document.getElementById('startBtn');"
    "const stopBtn = document.getElementById('stopBtn');"
    "const blockBtn = document.getElementById('blockBtn');"
    "const unblockBtn = document.getElementById('unblockBtn');"
    "const ifaceInput = document.getElementById('iface');"
    "const tableBody = document.querySelector('#alertsTable tbody');"
    "const statusDiv = document.getElementById('status');"
    "let selectedRow = null;"
    "let ws;"
    "function connectWebSocket() {"
    "  ws = new WebSocket('ws://' + location.host + '/ws');"
    "  ws.onopen = () => updateStatus('WebSocket connected. Ready.', 'green');"
    "  ws.onclose = () => { updateStatus('WebSocket disconnected. Retrying...', 'red'); setTimeout(connectWebSocket, 3000); };"
    "  ws.onmessage = (event) => {"
    "    const alert = JSON.parse(event.data);"
    "    const row = tableBody.insertRow(0);"
    "    row.innerHTML = `<td>${alert.timestamp}</td><td>${alert.type}</td><td>${alert.src_ip}</td><td>${alert.dest_ip}</td><td>${alert.details}</td>`;"
    "    if (alert.type.toLowerCase().includes('error')) row.style.color = 'red';"
    "    if (alert.type.toLowerCase().includes('firewall')) row.style.color = 'orange';"
    "    if (tableBody.rows.length > 200) tableBody.deleteRow(-1);"
    "  };"
    "}"
    "function updateStatus(message, color) { statusDiv.textContent = 'Status: ' + message; statusDiv.style.color = color || '#f1f1f1'; }"
    "startBtn.addEventListener('click', () => {"
    "  fetch('/api/start', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ iface: ifaceInput.value }) })"
    "    .then(res => res.json()).then(data => updateStatus(data.message));"
    "  tableBody.innerHTML = '';"
    "});"
    "stopBtn.addEventListener('click', () => {"
    "  fetch('/api/stop', { method: 'POST' })"
    "    .then(res => res.json()).then(data => updateStatus(data.message));"
    "});"
    "tableBody.addEventListener('click', (e) => {"
    "  const row = e.target.closest('tr');"
    "  if (!row) return;"
    "  if (selectedRow) selectedRow.classList.remove('selected');"
    "  selectedRow = row;"
    "  selectedRow.classList.add('selected');"
    "});"
    "function handleFirewallAction(action) {"
    "  if (!selectedRow) { alert('Please select an alert from the table first.'); return; }"
    "  const ip = selectedRow.cells[2].textContent;"
    "  if (!ip || ip === 'System') { alert('No valid source IP selected.'); return; }"
    "  fetch(`/api/${action}`, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ ip: ip }) })"
    "    .then(res => res.json()).then(data => updateStatus(data.message));"
    "  selectedRow.classList.remove('selected'); selectedRow = null;"
    "}"
    "blockBtn.addEventListener('click', () => handleFirewallAction('block'));"
    "unblockBtn.addEventListener('click', () => handleFirewallAction('unblock'));"
    "window.onload = connectWebSocket;"
    "</script>"
    "</body>"
    "</html>";

// --- Mongoose Web Server Event Handler (UPDATED) ---
static void fn(struct mg_connection *c, int ev, void *ev_data) {
    if (ev == MG_EV_HTTP_MSG) {
        struct mg_http_message *hm = (struct mg_http_message *) ev_data;

        if (mg_match(hm->uri, mg_str("/"), NULL)) {
             mg_http_reply(c, 200, "Content-Type: text/html\r\n", s_html_template);

        } else if (mg_match(hm->uri, mg_str("/api/start"), NULL)) {
            if (g_is_sniffer_running) {
                mg_http_reply(c, 400, "Content-Type: application/json\r\n", "{\"message\": \"Sniffer already running\"}");
                return;
            }
            // Updated JSON parsing for modern Mongoose
            char *iface_str = mg_json_get_str(hm->body, "$.iface");
            if (iface_str && strlen(iface_str) > 0) {
                pthread_t sniffer_thread;
                g_is_sniffer_running = 1;
                // strdup because iface_str will be freed
                pthread_create(&sniffer_thread, NULL, sniffer_thread_func, strdup(iface_str));
                pthread_detach(sniffer_thread);
                mg_http_reply(c, 200, "Content-Type: application/json\r\n", "{\"message\": \"Sniffer start command issued\"}");
            } else {
                mg_http_reply(c, 400, "Content-Type: application/json\r\n", "{\"message\": \"Interface name is required\"}");
            }
            if (iface_str) free(iface_str); // Important: free the string returned by mg_json_get_str

        } else if (mg_match(hm->uri, mg_str("/api/stop"), NULL)) {
            if (g_is_sniffer_running && g_pcap_handle != NULL) {
                pcap_breakloop(g_pcap_handle);
                mg_http_reply(c, 200, "Content-Type: application/json\r\n", "{\"message\": \"Sniffer stop command issued\"}");
            } else {
                mg_http_reply(c, 400, "Content-Type: application/json\r\n", "{\"message\": \"Sniffer not running\"}");
            }

        } else if (mg_match(hm->uri, mg_str("/api/block"), NULL) || mg_match(hm->uri, mg_str("/api/unblock"), NULL)) {
            char *ip_str = mg_json_get_str(hm->body, "$.ip");
            if (ip_str) {
                char command[256];
                const char *action = mg_match(hm->uri, mg_str("/api/block"), NULL) ? "-A" : "-D";
                snprintf(command, sizeof(command), "iptables %s INPUT -s %s -j DROP", action, ip_str);

                if (system(command) == 0) {
                    log_and_broadcast_alert("Firewall", "System", ip_str, "IP address action successful.");
                    mg_http_reply(c, 200, "Content-Type: application/json\r\n", "{\"message\": \"Firewall rule updated for %s\"}", ip_str);
                } else {
                    log_and_broadcast_alert("Firewall Error", "System", ip_str, "Failed to update firewall. Are you root?");
                    mg_http_reply(c, 500, "Content-Type: application/json\r\n", "{\"message\": \"Failed to update firewall for IP %s\"}", ip_str);
                }
                free(ip_str); // Free the JSON string
            } else {
                 mg_http_reply(c, 400, "Content-Type: application/json\r\n", "{\"message\": \"IP address required\"}");
            }

        } else if (mg_match(hm->uri, mg_str("/ws"), NULL)) {
             mg_ws_upgrade(c, hm, NULL);

        } else {
            mg_http_reply(c, 404, "", "{\"error\": \"Not Found\"}");
        }
    } else if (ev == MG_EV_WS_OPEN) {
        g_ws_connection = c;
        log_and_broadcast_alert("Info", "System", "N/A", "Web client connected.");
    } else if (ev == MG_EV_CLOSE) {
        if (c == g_ws_connection) {
            g_ws_connection = NULL;
            printf("Web client disconnected.\n");
        }
    }
}

// --- Main Function ---
int main(void) {
    mg_mgr_init(&g_mgr);
    printf("Starting web server on http://127.0.0.1:8000\n");
    // Note: modern Mongoose often uses mg_http_listen with 4 args, but if your version
    // complained about standard 'fn' having 4 args, it likely wants the 3-arg version here too.
    // If this still fails, try removing ", NULL" from the end of this call.
    mg_http_listen(&g_mgr, "http://0.0.0.0:8000", fn, NULL);

    while (1) {
        mg_mgr_poll(&g_mgr, 1000);
    }

    mg_mgr_free(&g_mgr);
    return 0;
}

// --- Helper Functions Implementation ---
int find_ip_state(in_addr_t ip) {
    for (int i = 0; i < current_tracked_ips; i++) {
        if (ip_states[i].ip_addr == ip) return i;
    }
    return -1;
}

void add_unique_port(int state_idx, unsigned short port) {
    for (int i = 0; i < ip_states[state_idx].port_count; i++) {
        if (ip_states[state_idx].unique_ports[i] == port) return;
    }
    if (ip_states[state_idx].port_count < MAX_UNIQUE_PORTS_PER_IP) {
        ip_states[state_idx].unique_ports[ip_states[state_idx].port_count++] = port;
    }
}

void cleanup_old_states() {
    time_t now = time(NULL);
    for (int i = 0; i < current_tracked_ips; i++) {
        if (difftime(now, ip_states[i].last_activity) > SCAN_TIME_WINDOW_SEC) {
            ip_states[i] = ip_states[current_tracked_ips - 1];
            current_tracked_ips--;
            i--;
        }
    }
}
