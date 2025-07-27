import tkinter as tk
from tkinter import messagebox
import random

class MemoryGame:
    def __init__(self, master):
        self.master = master
        self.master.title("Memory Match Game")
        self.master.configure(bg='#f0f8ff')
        self.levels_frame()

    def levels_frame(self):
        self.clear_frame()
        tk.Label(self.master, text="Select Game Level", font=("Arial", 24), bg='#f0f8ff', fg='#2e8b57').pack(pady=20)
        tk.Button(self.master, text="Easy (4x4)", command=lambda: self.players_frame(4), bg='#add8e6', width=15, height=2).pack(pady=10)
        tk.Button(self.master, text="Medium (6x6)", command=lambda: self.players_frame(6), bg='#add8e6', width=15, height=2).pack(pady=10)
        tk.Button(self.master, text="Hard (8x8)", command=lambda: self.players_frame(8), bg='#add8e6', width=15, height=2).pack(pady=10)

    def players_frame(self, grid_size):
        self.clear_frame()
        self.grid_size = grid_size
        tk.Label(self.master, text="Select Player Mode", font=("Arial", 24), bg='#f0f8ff', fg='#2e8b57').pack(pady=20)
        tk.Button(self.master, text="Single Player", command=lambda: self.name_entry_frame(1), bg='#add8e6').pack(pady=10)
        tk.Button(self.master, text="Two Players", command=lambda: self.name_entry_frame(2), bg='#add8e6').pack(pady=10)

    def name_entry_frame(self, num_players):
        self.clear_frame()
        self.num_players = num_players

        tk.Label(self.master, text="Enter Player 1 Name:", font=("Arial", 18), bg='#f0f8ff', fg='#2e8b57').pack(pady=10)
        self.player1_name_entry = tk.Entry(self.master)
        self.player1_name_entry.pack(pady=10)

        if num_players == 2:
            tk.Label(self.master, text="Enter Player 2 Name:", font=("Arial", 18), bg='#f0f8ff', fg='#2e8b57').pack(pady=10)
            self.player2_name_entry = tk.Entry(self.master)
            self.player2_name_entry.pack(pady=10)

        tk.Button(self.master, text="Start Game", command=self.start_game, bg='#add8e6').pack(pady=20)

    def start_game(self):
        if not self.player1_name_entry.get():
            messagebox.showerror("Error", "Enter Player 1 Name")
            return
        if self.num_players == 2 and not self.player2_name_entry.get():
            messagebox.showerror("Error", "Enter Player 2 Name")
            return

        self.name1 = self.player1_name_entry.get()
        self.name2 = self.player2_name_entry.get() if self.num_players == 2 else 'Computer'

        self.score1 = 0
        self.score2 = 0
        self.player_turn = 1
        self.flipped_cards = []
        self.memory = {}
        self.cards = self.setup_cards()
        self.game_frame()

    def setup_cards(self):
        pairs = ["🍦", "🍫", "🎂", "🍎", "🥕", "😀", "🍕", "🥩", "🍩", "🍇", "🍌", "🍓", "🥑", "🍉", "🧁", "🍒",
                 "🥥", "🍍", "🍔", "🍟", "🌮", "🍪", "🍵", "🥞", "🧀", "🍗", "🍖", "🍰", "🍛", "🍤", "🍣", "🥗",
                 "🥬", "🍜", "🥮", "🍙", "🧇", "🍿", "🍠", "🍞"]
        needed_pairs = self.grid_size * self.grid_size // 2
        card_faces = random.sample(pairs, needed_pairs) * 2
        random.shuffle(card_faces)
        return [card_faces[i * self.grid_size:(i + 1) * self.grid_size] for i in range(self.grid_size)]

    def game_frame(self):
        self.clear_frame()
        self.labels = []
        for i in range(self.grid_size):
            row = []
            for j in range(self.grid_size):
                label = tk.Label(self.master, text='', width=4, height=2, bg='cyan', font=("Arial", 12), relief="raised", borderwidth=1)
                label.grid(row=i, column=j, padx=5, pady=5)
                label.bind("<Button-1>", lambda e, pos=(i, j): self.flip_card(pos))
                row.append(label)
            self.labels.append(row)

        self.score_label = tk.Label(self.master, text=self.get_score_text(), font=("Arial", 14), bg='#f0f8ff', fg='#2e8b57')
        self.score_label.grid(row=self.grid_size, columnspan=self.grid_size)

        self.current_player_label = tk.Label(self.master, text=f"Current Turn: {self.name1}", font=("Arial", 14), bg='#f0f8ff', fg='#2e8b57')
        self.current_player_label.grid(row=self.grid_size + 1, columnspan=self.grid_size)

        if self.player_turn == 2 and self.name2 == 'Computer':
            self.master.after(1000, self.computer_turn)

    def flip_card(self, pos):
        x, y = pos
        label = self.labels[x][y]
        if label['text'] == '' and len(self.flipped_cards) < 2:
            label['text'] = self.cards[x][y]
            self.flipped_cards.append((label, (x, y)))

            if len(self.flipped_cards) == 2:
                self.master.after(1000, self.check_match)

    def check_match(self):
        (label1, pos1), (label2, pos2) = self.flipped_cards
        val1, val2 = self.cards[pos1[0]][pos1[1]], self.cards[pos2[0]][pos2[1]]

        if val1 == val2:
            if self.player_turn == 1:
                self.score1 += 1
            else:
                self.score2 += 1
        else:
            label1['text'] = ''
            label2['text'] = ''

            # Store the positions for future reference
            self.memory[val1] = self.memory.get(val1, []) + [pos1]
            self.memory[val2] = self.memory.get(val2, []) + [pos2]

        self.flipped_cards = []
        self.score_label['text'] = self.get_score_text()

        if self.check_game_over():
            self.show_winner()
            return

        self.player_turn = 2 if self.player_turn == 1 else 1
        self.current_player_label['text'] = f"Current Turn: {self.name1 if self.player_turn == 1 else self.name2}"

        if self.player_turn == 2 and self.name2 == 'Computer':
            self.master.after(1000, self.computer_turn)

    def computer_turn(self):
        unseen = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size) if self.labels[i][j]['text'] == '']
        if len(unseen) < 2:
            return

        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for i, pos1 in enumerate(unseen):
            for pos2 in unseen[i + 1:]:
                new_score = self.simulate_move(pos1, pos2)
                if new_score > best_score:
                    best_score = new_score
                    best_move = (pos1, pos2)

                # Alpha-Beta pruning logic
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break

        if best_move:
            pos1, pos2 = best_move
            self.flip_card(pos1)
            self.master.after(700, lambda p=pos2: self.flip_card(p))

    def simulate_move(self, pos1, pos2):
        # Simulate the move and return score
        x1, y1 = pos1
        x2, y2 = pos2
        val1 = self.cards[x1][y1]
        val2 = self.cards[x2][y2]
        return 1 if val1 == val2 else 0

    def check_game_over(self):
        return all(label['text'] != '' for row in self.labels for label in row)

    def show_winner(self):
        if self.score1 > self.score2:
           winner_text = f"The winner is: {self.name1}"
        elif self.score2 > self.score1:
            winner_text = f"The winner is: {self.name2}"
        else:
            winner_text = "It's a Draw!"

        score_summary = f"{self.name1}: {self.score1} - {self.name2}: {self.score2}"
        self.clear_frame()
        tk.Label(self.master, text="Game Over!", font=("Arial", 24), bg='#f0f8ff', fg='#2e8b57').pack(pady=20)
        tk.Label(self.master, text=winner_text, font=("Arial", 18), bg='#f0f8ff', fg='#2e8b57').pack(pady=10)
        tk.Label(self.master, text=score_summary, font=("Arial", 16), bg='#f0f8ff', fg='#2e8b57').pack(pady=10)
        tk.Button(self.master, text="Back to Main Menu", font=("Arial", 14), bg='#add8e6', command=self.levels_frame).pack(pady=20)

    def get_score_text(self):
        return f"{self.name1}'s Score: {self.score1}  |  {self.name2}'s Score: {self.score2}"

    def clear_frame(self):
        for widget in self.master.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    game = MemoryGame(root)
    root.mainloop()











