import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './index.css';

// Base URL for your Flask backend
const API_URL = "http://127.0.0.1:5000/api";

// Helper for UI
const highlightDifferences = (ref, hyp) => {
  // FIX: Added a guard clause to handle cases where the text is empty or null.
  if (!ref || !hyp) {
    return "";
  }
  const refWords = ref.trim().split(/\s+/);
  const hypWords = hyp.trim().split(/\s+/);
  const matcher = new SequenceMatcher(null, refWords, hypWords);
  const html = [];
  matcher.getOpcodes().forEach(([tag, i1, i2, j1, j2]) => {
    if (tag === 'equal') {
      refWords.slice(i1, i2).forEach(w => html.push(`<span class='correct-word'>${w}</span>`));
    } else if (tag === 'replace') {
      refWords.slice(i1, i2).forEach(w => html.push(`<span class='incorrect-word'>${w}</span>`));
      html.push(" → ");
      hypWords.slice(j1, j2).forEach(w => html.push(`<span class='user-word'>${w}</span>`));
    } else if (tag === 'delete') {
      refWords.slice(i1, i2).forEach(w => html.push(`<span class='missing-word'>${w}</span>`));
    } else if (tag === 'insert') {
      html.push(" + ");
      hypWords.slice(j1, j2).forEach(w => html.push(`<span class='extra-word'>${w}</span>`));
    }
  });
  return html.join(" ");
};

function SequenceMatcher(isjunk, a, b) {
  this.a = a;
  this.b = b;
  this.opcodes = [];
  const matrix = Array(a.length + 1).fill(null).map(() => Array(b.length + 1).fill(0));
  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      if (a[i - 1] === b[j - 1]) {
        matrix[i][j] = matrix[i - 1][j - 1] + 1;
      } else {
        matrix[i][j] = Math.max(matrix[i - 1][j], matrix[i][j - 1]);
      }
    }
  }
  let i = a.length, j = b.length;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && a[i - 1] === b[j - 1]) {
      let k = 1;
      while (i - k > 0 && j - k > 0 && a[i - 1 - k] === b[j - 1 - k]) k++;
      this.opcodes.unshift(['equal', i - k, i, j - k, j]);
      i -= k;
      j -= k;
    } else if (j > 0 && (i === 0 || matrix[i][j - 1] >= matrix[i - 1][j])) {
      this.opcodes.unshift(['insert', i, i, j - 1, j]);
      j--;
    } else if (i > 0 && (j === 0 || matrix[i][j - 1] < matrix[i - 1][j])) {
      this.opcodes.unshift(['delete', i - 1, i, j, j]);
      i--;
    } else {
      let k = 1;
      while (i - k > 0 && j - k > 0 && a[i - 1 - k] !== b[j - 1 - k]) k++;
      this.opcodes.unshift(['replace', i - k, i, j - k, j]);
      i -= k;
      j -= k;
    }
  }
  // FIX: Added this method to make the function callable
  this.getOpcodes = () => this.opcodes;
};

// =========================================================================================

const AuthPage = ({ setUser }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [message, setMessage] = useState('');
  const [isError, setIsError] = useState(false);

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.post(`${API_URL}/login`, { username, password });
      setUser(res.data.user);
    } catch (err) {
      setMessage(err.response?.data?.message || "Login failed.");
      setIsError(true);
    }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.post(`${API_URL}/register`, { username, password, full_name: fullName });
      setMessage(res.data.message);
      setIsError(false);
      setIsLogin(true); // Switch to login tab after successful registration
    } catch (err) {
      setMessage(err.response?.data?.message || "Registration failed.");
      setIsError(true);
    }
  };

  return (
    <div style={{ padding: '2rem' }}>
      <div className="hero">
        <h1 style={{ margin: 0, fontSize: '2.5rem' }}>🎤 FluentMe</h1>
        <p style={{ margin: 0, fontSize: '1.2rem', opacity: 0.9 }}>Speak confidently with our speech therapy coach</p>
      </div>
      <div className="login-container">
        <img src="/pexels-cristian-rojas-7586662.jpg" width="60%" style={{ borderRadius: '16px' }} alt="login visual" />
        <div className="card" style={{ flex: 1 }}>
          <div className="tab-nav">
            <button onClick={() => setIsLogin(true)} className={isLogin ? 'active' : ''}>Login</button>
            <button onClick={() => setIsLogin(false)} className={!isLogin ? 'active' : ''}>Register</button>
          </div>
          {message && <div style={{ color: isError ? 'red' : 'green' }}>{message}</div>}
          {isLogin ? (
            <form onSubmit={handleLogin} className="login-form">
              <h3>Welcome Back!</h3>
              <input type="text" placeholder="Username" value={username} onChange={e => setUsername(e.target.value)} required />
              <input type="password" placeholder="Password" value={password} onChange={e => setPassword(e.target.value)} required />
              <button type="submit" className="btn btn-primary btn-block">Login</button>
            </form>
          ) : (
            <form onSubmit={handleRegister} className="login-form">
              <h3>Create Account</h3>
              <input type="text" placeholder="Choose a username" value={username} onChange={e => setUsername(e.target.value)} required />
              <input type="password" placeholder="Choose a password" value={password} onChange={e => setPassword(e.target.value)} required />
              <input type="text" placeholder="Full name (optional)" value={fullName} onChange={e => setFullName(e.target.value)} />
              <button type="submit" className="btn btn-primary btn-block">Register</button>
            </form>
          )}
        </div>
      </div>
    </div>
  );
};

// =========================================================================================

const Sidebar = ({ user, onLogout, onTabChange, setConfig }) => {
  const [puzzle, setPuzzle] = useState(null);
  const [unscrambledWord, setUnscrambledWord] = useState('');
  const [solved, setSolved] = useState(false);

  useEffect(() => {
    const fetchPuzzle = async () => {
      try {
        const res = await axios.get(`${API_URL}/daily-puzzle?userId=${user.id}`);
        setPuzzle(res.data.puzzle);
        setUnscrambledWord(res.data.unscrambled);
        setSolved(res.data.solved);
      } catch (err) {
        console.error("Failed to fetch puzzle:", err);
      }
    };
    fetchPuzzle();
  }, [user]);

  const handleSolve = async () => {
    try {
      await axios.post(`${API_URL}/solve-puzzle`, { userId: user.id });
      setSolved(true);
      alert("Puzzle solved! Great job.");
    } catch (err) {
      console.error("Failed to solve puzzle:", err);
    }
  };

  return (
    <div className="sidebar">
      <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
        <h2>🎯 Daily Puzzle</h2>
      </div>
      <div className="puzzle-card">
        <h3 style={{ margin: 0, color: '#333' }}>{solved ? unscrambledWord.toUpperCase() : "Solve this puzzle!"}</h3>
        <p className="scrambled-word">{solved ? "" : puzzle}</p>
        <p style={{ margin: '0.5rem 0', color: '#666' }}>
          {solved ? "Puzzle solved!" : "Unscramble the letters to guess the word."}
        </p>
      </div>
      {!solved && <button className="btn btn-primary btn-block" onClick={handleSolve}>✅ Mark as Solved</button>}
      <hr />
      <div style={{ textAlign: 'center' }}>
        <h2>⚙️ Settings</h2>
      </div>
      <div className="form-group">
        <label>Practice Language</label>
        <select onChange={(e) => setConfig(prev => ({ ...prev, language: e.target.value }))}>
          <option value="en">English</option>
          <option value="hi">Hindi</option>
          <option value="es">Spanish</option>
          <option value="fr">French</option>
        </select>
      </div>
      <div className="form-group">
        <label>Whisper Model</label>
        <select onChange={(e) => setConfig(prev => ({ ...prev, model: e.target.value }))}>
          <option value="tiny.en">tiny.en</option>
          <option value="tiny">tiny</option>
          <option value="base.en">base.en</option>
          <option value="base">base</option>
        </select>
      </div>
      <div className="form-group">
        <label>Pass Threshold</label>
        <input type="range" min="50" max="95" defaultValue="80" onChange={e => setConfig(prev => ({ ...prev, threshold: e.target.value / 100 }))} />
        <span>{Math.round(localStorage.getItem('threshold') * 100) || 80}%</span>
      </div>
      <button className="btn btn-secondary btn-block" onClick={onLogout}>🚪 Logout</button>
    </div>
  );
};

// =========================================================================================

const HomePage = ({ onAction }) => (
  <div className="tab-content">
    <div className="hero">
      <h2>Welcome to FluentMe</h2>
      <p>Your personal speech therapy coach</p>
    </div>
    <div className="card">
      <h3>Get Started</h3>
      <p>Record or upload an audio clip, compare it to a sentence, and track your improvements over time.</p>
    </div>
    <div className="flex-container">
      <div style={{ flex: 1 }}><img src="https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?q=80&w=1200&auto=format&fit=crop" style={{ width: '100%', borderRadius: '16px' }} alt="practice" /></div>
      <div style={{ flex: 1 }}><img src="/smiling-cartoon-speech-white-background_53876-486536.jpg" style={{ width: '100%', borderRadius: '16px' }} alt="play" /></div>
    </div>
    <hr />
    <h3>Quick Actions</h3>
    <div className="flex-container">
      <div style={{ flex: 1 }}>
        <div className="quick-action-btn" onClick={() => onAction('practice', 'cat')}>
          <h4>🎯 Beginner Task</h4>
          <p>Try a simple pronunciation exercise</p>
        </div>
      </div>
      <div style={{ flex: 1 }}>
        <div className="quick-action-btn" onClick={() => onAction('tts', "Welcome to FluentMe. Speak after me.")}>
          <h4>🔊 TTS Example</h4>
          <p>Listen to an example text-to-speech</p>
        </div>
      </div>
      <div style={{ flex: 1 }}>
        <div className="quick-action-btn" onClick={() => onAction('tips')}>
          <h4>💡 Tips</h4>
          <p>Get pronunciation tips</p>
        </div>
      </div>
    </div>
  </div>
);

const PracticePage = ({ user, config }) => {
  const [refText, setRefText] = useState("The brown fox jumped over the gate");
  const [file, setFile] = useState(null);
  const [transcript, setTranscript] = useState('');
  const [accuracy, setAccuracy] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [playbackAudio, setPlaybackAudio] = useState(null);

  const handleAudioUpload = async (e) => {
    const uploadedFile = e.target.files[0];
    if (!uploadedFile) return;

    setFile(uploadedFile);
    setLoading(true);
    setMessage('');

    const formData = new FormData();
    formData.append('audio', uploadedFile);
    formData.append('referenceText', refText);
    formData.append('userId', user.id);
    formData.append('language', config.language);
    formData.append('model', config.model);

    try {
      const res = await axios.post(`${API_URL}/practice`, formData, { headers: { 'Content-Type': 'multipart/form-data' } });
      setTranscript(res.data.transcript);
      setAccuracy(res.data.accuracy);
      if (res.data.accuracy >= config.threshold) {
        setMessage('🎉 Great job!');
      } else {
        setMessage('Keep practicing!');
      }
      
      const recordProgress = await axios.post(`${API_URL}/record-progress`, {
        userId: user.id,
        level: "Practice",
        task: refText,
        passed: res.data.accuracy >= config.threshold,
        accuracy: res.data.accuracy
      });

    } catch (err) {
      setMessage('An error occurred during transcription.');
      setAccuracy(null);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handlePreviewTTS = async () => {
    try {
      const res = await axios.post(`${API_URL}/tts`, { text: refText, lang: config.language }, { responseType: 'blob' });
      setPlaybackAudio(URL.createObjectURL(res.data));
    } catch (err) {
      console.error("Failed to generate TTS:", err);
    }
  };

  return (
    <div className="tab-content">
      <div className="hero">
        <h2>Practice Mode</h2>
        <p>Speak, compare, and improve your pronunciation</p>
      </div>
      <div className="form-group">
        <label>Text to practice</label>
        <textarea value={refText} onChange={e => setRefText(e.target.value)} rows="4" placeholder="Type or paste the text you want to practice..."></textarea>
      </div>
      <div className="flex-container">
        <div style={{ flex: 2 }}>
          <div className="card">
            <h3>Record or Upload</h3>
            <p>Upload your audio file to get a fluency score.</p>
            <input type="file" accept="audio/*" onChange={handleAudioUpload} />
            {loading && <p>Transcribing audio...</p>}
            {accuracy !== null && (
              <div>
                <h4>Results:</h4>
                <p>{message} Accuracy: {(accuracy * 100).toFixed(1)}%</p>
                <div style={{ padding: '1rem', background: '#f8f9fa', borderRadius: '12px' }}
                  dangerouslySetInnerHTML={{ __html: highlightDifferences(refText, transcript) }} />
                {playbackAudio && <audio controls src={playbackAudio} />}
              </div>
            )}
          </div>
        </div>
        <div style={{ flex: 1 }}>
          <div className="card">
            <h3>Preview & Actions</h3>
            <button className="btn btn-primary btn-block" onClick={handlePreviewTTS}>Preview TTS</button>
            {playbackAudio && <audio controls src={playbackAudio} />}
            <hr />
            <h4>Tips for better results:</h4>
            <ul>
              <li>Speak clearly and at a moderate pace</li>
              <li>Reduce background noise</li>
              <li>Hold the microphone close to your mouth</li>
              <li>Practice in a quiet environment</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

const GameModePage = ({ user, config }) => {
  const [level, setLevel] = useState('Beginner');
  const [task, setTask] = useState('');
  const [audioFile, setAudioFile] = useState(null);
  const [transcript, setTranscript] = useState('');
  const [accuracy, setAccuracy] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [gameTasks, setGameTasks] = useState({});
  const [playbackAudio, setPlaybackAudio] = useState(null);

  useEffect(() => {
    const fetchTasks = async () => {
      const res = await axios.get(`${API_URL}/game-tasks`);
      setGameTasks(res.data);
      if (res.data[level]) {
        setTask(res.data[level][Math.floor(Math.random() * res.data[level].length)]);
      }
    };
    fetchTasks();
  }, [level]);

  const handleNewChallenge = () => {
    if (gameTasks[level]) {
      setTask(gameTasks[level][Math.floor(Math.random() * gameTasks[level].length)]);
      setTranscript('');
      setAccuracy(null);
      setAudioFile(null);
      setPlaybackAudio(null);
    }
  };

  const handleAudioUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setAudioFile(file);
    setLoading(true);

    const formData = new FormData();
    formData.append('audio', file);
    formData.append('referenceText', task);
    formData.append('userId', user.id);
    formData.append('language', config.language);
    formData.append('model', config.model);

    try {
      const res = await axios.post(`${API_URL}/practice`, formData, { headers: { 'Content-Type': 'multipart/form-data' } });
      setTranscript(res.data.transcript);
      setAccuracy(res.data.accuracy);
      if (res.data.accuracy >= config.threshold) {
        setMessage('🎉 Excellent!');
      } else {
        setMessage('Good attempt! Keep practicing.');
      }
      
      await axios.post(`${API_URL}/record-progress`, {
        userId: user.id,
        level: level,
        task: task,
        passed: res.data.accuracy >= config.threshold,
        accuracy: res.data.accuracy
      });
    } catch (err) {
      setMessage('An error occurred.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handlePlayTTS = async () => {
    try {
      const res = await axios.post(`${API_URL}/tts`, { text: task, lang: config.language }, { responseType: 'blob' });
      setPlaybackAudio(URL.createObjectURL(res.data));
    } catch (err) {
      console.error("Failed to generate TTS:", err);
    }
  };

  return (
    <div className="tab-content">
      <div className="hero">
        <h2>Game Mode</h2>
        <p>Fun challenges to improve your pronunciation</p>
      </div>
      <div className="form-group">
        <label>Choose difficulty level</label>
        <select value={level} onChange={e => setLevel(e.target.value)}>
          {Object.keys(gameTasks).map(l => <option key={l} value={l}>{l}</option>)}
        </select>
      </div>
      <div className="card">
        <h3>Your Challenge:</h3>
        <p style={{ fontSize: '1.5rem', fontWeight: 'bold', textAlign: 'center', padding: '1rem', background: '#f8f9fa', borderRadius: '12px' }}>{task}</p>
      </div>
      <div className="flex-container">
        <button className="btn btn-secondary" onClick={handlePlayTTS}>🎧 Play Example</button>
        <button className="btn btn-secondary" onClick={handleNewChallenge}>🔄 New Challenge</button>
      </div>
      <hr />
      <h3>Record Your Attempt</h3>
      <input type="file" accept="audio/*" onChange={handleAudioUpload} />
      {loading && <p>Evaluating your attempt...</p>}
      {accuracy !== null && (
        <div>
          <p>{message} You scored {(accuracy * 100).toFixed(1)}%</p>
          <h4>Your transcription:</h4>
          <p style={{ padding: '1rem', background: '#f0f0f0', borderRadius: '10px' }}>{transcript}</p>
          {audioFile && <audio controls src={URL.createObjectURL(audioFile)} />}
        </div>
      )}
      {playbackAudio && <audio controls src={playbackAudio} />}
    </div>
  );
};

const LeaderboardPage = () => {
  const [leaderboard, setLeaderboard] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchLeaderboard = async () => {
      try {
        const res = await axios.get(`${API_URL}/leaderboard`);
        setLeaderboard(res.data);
      } catch (err) {
        console.error("Failed to fetch leaderboard:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchLeaderboard();
  }, []);

  if (loading) return <p>Loading leaderboard...</p>;

  return (
    <div className="tab-content">
      <div className="hero">
        <h2>Leaderboard</h2>
        <p>See how you compare to other users</p>
      </div>
      {leaderboard.length > 0 ? (
        leaderboard.map((item, index) => (
          <div key={index} className="leaderboard-item">
            <div>
              <span className="leaderboard-rank">#{index + 1}</span>
              <strong>{item.username}</strong>
            </div>
            <div style={{ color: '#667eea', fontWeight: 'bold' }}>{(item.avg_acc * 100).toFixed(1)}%</div>
          </div>
        ))
      ) : (
        <p>No attempts yet. Be the first to practice!</p>
      )}
    </div>
  );
};

const ProgressPage = ({ user }) => {
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const historyRes = await axios.get(`${API_URL}/progress/${user.id}`);
        setHistory(historyRes.data);
        const statsRes = await axios.get(`${API_URL}/stats/${user.id}`);
        setStats(statsRes.data);
      } catch (err) {
        console.error("Failed to fetch progress:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [user]);

  if (loading) return <p>Loading progress data...</p>;

  return (
    <div className="tab-content">
      <div className="hero">
        <h2>Your Progress</h2>
        <p>Track your improvement over time</p>
      </div>
      <div className="metrics-row">
        <div className="card"><h3>{stats.attempts_count}</h3><p>Total Attempts</p></div>
        <div className="card"><h3>{(stats.latest_acc * 100).toFixed(1)}%</h3><p>Latest Accuracy</p></div>
        <div className="card"><h3>{stats.streak}/7</h3><p>Weekly Streak</p></div>
      </div>
      <h3>Accuracy Trend</h3>
      {history.length > 0 ? (
        <p>You can visualize your progress here. (Note: A proper chart library would be needed for visual display).</p>
      ) : (
        <p>No data yet. Start practicing to see your progress.</p>
      )}
      <h3>Recent Attempts</h3>
      {history.slice(-5).reverse().map((attempt, index) => (
        <div key={index} className="card">
          <p>Accuracy: {(attempt.accuracy * 100).toFixed(1)}%</p>
          <p>Date: {new Date(attempt.created_at).toLocaleDateString()}</p>
        </div>
      ))}
    </div>
  );
};

// =========================================================================================

const App = () => {
  const [user, setUser] = useState(JSON.parse(localStorage.getItem('user')));
  const [activeTab, setActiveTab] = useState('Home');
  const [config, setConfig] = useState({
    language: 'en',
    model: 'tiny.en',
    threshold: 0.8
  });

  const handleLogout = () => {
    localStorage.removeItem('user');
    setUser(null);
  };

  const handleQuickAction = (action, payload) => {
    if (action === 'practice') {
      setActiveTab('Practice');
    } else if (action === 'tts') {
      const audio = new Audio();
      const playTTS = async () => {
        try {
          const res = await axios.post(`${API_URL}/tts`, { text: payload, lang: config.language }, { responseType: 'blob' });
          audio.src = URL.createObjectURL(res.data);
          audio.play();
        } catch (err) {
          console.error("Failed to play TTS:", err);
        }
      };
      playTTS();
    } else if (action === 'tips') {
      alert("Tips:\n- Speak slowly and clearly\n- Focus on vowel sounds\n- Use headphones for better playback\n- Practice in a quiet environment\n- Record yourself regularly to track progress");
    }
  };

  useEffect(() => {
    if (user) {
      localStorage.setItem('user', JSON.stringify(user));
    }
  }, [user]);

  if (!user) {
    return <AuthPage setUser={setUser} />;
  }

  return (
    <div className="App">
      <Sidebar user={user} onLogout={handleLogout} setConfig={setConfig} />
      <div className="main-content">
        <div className="header">
          <img src={"/3d-cartoon-avatar-girl-minimal-3d-character_652053-2350.jpg"} width="80" style={{ borderRadius: '50%' }} alt="user avatar" />
          <div>
            <h2>Welcome, {user.full_name || user.username} 👋</h2>
            <p>Practice daily to build consistent improvement — just 10 minutes a day.</p>
          </div>
        </div>
        <div className="tab-nav">
          {['Home', 'Practice', 'Game Mode', 'Leaderboard', 'Progress'].map(tab => (
            <button key={tab} className={activeTab === tab ? 'active' : ''} onClick={() => setActiveTab(tab)}>
              {tab}
            </button>
          ))}
        </div>
        {activeTab === 'Home' && <HomePage onAction={handleQuickAction} />}
        {activeTab === 'Practice' && <PracticePage user={user} config={config} />}
        {activeTab === 'Game Mode' && <GameModePage user={user} config={config} />}
        {activeTab === 'Leaderboard' && <LeaderboardPage />}
        {activeTab === 'Progress' && <ProgressPage user={user} />}
      </div>
    </div>
  );
};

export default App;
