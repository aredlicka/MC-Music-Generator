from music21 import converter, instrument, stream, note, chord, environment
import numpy as np
from collections import defaultdict
import pandas as pd
from fractions import Fraction
import os

env = environment.Environment()
env['musicxmlPath'] = r'C:\Program Files\MuseScore 4\bin\MuseScore4.exe'  # ścieżka do MuseScore


class MCMusicGenerator:
    def __init__(self, training_data=None, cumulative_matrix_path=None, initial_cumulative_path=None):
        if cumulative_matrix_path and initial_cumulative_path:
            self.load_cumulative_matrix(cumulative_matrix_path)
            self.load_initial_cumulative(initial_cumulative_path)
        else:
            self.training_data = training_data
            self.unique_notes = []
            self.cumulative_matrix = None
            self.initial_cumulative = None
            self.process_training_data()

    def process_training_data(self):
        L_i = self.extract_notes_from_midi()  # Lista dźwięków
        self.create_transition_matrix(L_i)
        self.calculate_initial_probabilities(L_i)

    def extract_notes_from_midi(self):
        all_sequences = []
        for file_path in self.training_data:
            midi_data = converter.parse(file_path)
            parts = instrument.partitionByInstrument(midi_data)
            if parts:
                notes_to_parse = parts.parts[0].recurse()
            else:
                notes_to_parse = midi_data.flat.notes
            notes_sequence = []
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes_sequence.append((str(element.pitch), element.quarterLength))
                elif isinstance(element, chord.Chord):
                    notes_sequence.append(('.'.join(str(n) for n in element.pitches), element.quarterLength))

            if notes_sequence:
                notes_sequence.append(('C4', 1.0))  # Dodanie ćwierćnuty
                notes_sequence.append(notes_sequence[0])  # Dodanie przejścia do pierwszej nuty
            all_sequences.append(notes_sequence)
        return all_sequences

    def create_transition_matrix(self, sequences):
        note_counts = defaultdict(int)
        for seq in sequences:
            for note in seq:
                note_counts[note] += 1

        self.unique_notes = list(note_counts.keys())
        note_index = {note: i for i, note in enumerate(self.unique_notes)}
        n = len(self.unique_notes)
        transition_matrix = np.zeros((n, n))

        for seq in sequences:
            for i in range(len(seq) - 1):
                curr_note = seq[i]
                next_note = seq[i + 1]
                transition_matrix[note_index[curr_note], note_index[next_note]] += 1

        for i in range(n):
            total = np.sum(transition_matrix[i])
            if total > 0:
                transition_matrix[i] /= total
        self.cumulative_matrix = np.cumsum(transition_matrix, axis=1)

    def calculate_initial_probabilities(self, sequences):
        total_counts = {note: 0 for note in self.unique_notes}
        for seq in sequences:
            for note in seq:
                total_counts[note] += 1

        total_notes = sum(total_counts.values())
        initial_probabilities = np.array([total_counts[note] for note in self.unique_notes])
        initial_probabilities = initial_probabilities / total_notes
        self.initial_cumulative = np.cumsum(initial_probabilities)

    def save_cumulative_matrix(self, filename="cumulative_matrix.csv"):
        pd.DataFrame(self.cumulative_matrix, index=self.unique_notes, columns=self.unique_notes).to_csv(filename)

    def save_initial_cumulative(self, filename="initial_cumulative.csv"):
        pd.DataFrame(self.initial_cumulative, index=self.unique_notes).to_csv(filename)

    def load_cumulative_matrix(self, filename):
        df = pd.read_csv(filename, index_col=0)
        self.cumulative_matrix = df.to_numpy()
        safe_context = {'Fraction': Fraction}
        self.unique_notes = [eval(s, {"__builtins__": None}, safe_context) for s in df.columns]

    def load_initial_cumulative(self, filename):
        df = pd.read_csv(filename, index_col=0)
        self.initial_cumulative = df.iloc[:, 0].values

    def generate_music(self, num_notes=50):
        first_note_index = np.searchsorted(self.initial_cumulative, np.random.rand())
        music_sequence = [self.unique_notes[first_note_index]]

        current_note_index = first_note_index
        for _ in range(num_notes - 1):
            current_row = self.cumulative_matrix[current_note_index]
            next_note_index = np.searchsorted(current_row, np.random.rand())
            music_sequence.append(self.unique_notes[next_note_index])
            current_note_index = next_note_index

        return music_sequence

    def save_to_midi(self, notes_sequence, file_name="output.mid"):
        midi_stream = stream.Stream()
        for n, duration in notes_sequence:
            if '.' in n:  # it's a chord
                notes_in_chord = n.split('.')
                new_chord = chord.Chord(notes_in_chord)
                new_chord.quarterLength = duration
                midi_stream.append(new_chord)
            else:  # it's a single note
                new_note = note.Note(n)
                new_note.quarterLength = duration
                midi_stream.append(new_note)
        midi_stream.write('midi', fp=file_name)

    def show_score(self, notes_sequence):
        score_stream = stream.Stream()
        for n, duration in notes_sequence:
            if '.' in n:  # chord
                notes_in_chord = n.split('.')
                new_chord = chord.Chord(notes_in_chord)
                new_chord.quarterLength = duration
                score_stream.append(new_chord)
            else:  # note
                new_note = note.Note(n)
                new_note.quarterLength = duration
                score_stream.append(new_note)
        score_stream.show()


# Wywołanie

'''
folder_path = 'gaming'
files_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.mid', '.midi'))]
generator = MCMusicGenerator(files_list)

generated_music = generator.generate_music(100)
generator.save_to_midi(generated_music, "wygenerowane-gaming.mid")
#generator.show_score(generated_music)

#generator.save_cumulative_matrix("cumulative_matrix_gaming.csv")
#generator.save_initial_cumulative("initial_cumulative_gaming.csv")
'''

'''
matrix_path = r'macierze-i-wektory\cumulative_matrix_gaming.csv'  # UWAGA: W przypadku macierzy cumulative_matrix_giantmidi najpierw trzeba ją wypakować w folderze
initial_path = r'macierze-i-wektory\initial_cumulative_gaming.csv'
generator = MCMusicGenerator(cumulative_matrix_path=matrix_path, initial_cumulative_path=initial_path)

generated_music = generator.generate_music(100)
generator.save_to_midi(generated_music, "wygenerowane-gaming.mid")
#generator.show_score(generated_music)
'''
