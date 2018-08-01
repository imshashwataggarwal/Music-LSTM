from __future__ import division, print_function, absolute_import

import os, glob, random
import pretty_midi as Midi
import h5py
import numpy as np

from keras.models import model_from_json
from multiprocessing import Pool as ThreadPool

THRESHOLD = 1.0

# Save Keras Model
def save_model(model, path):

    model_json = model.to_json()
    with open(os.path.join(path,'model.json'), 'w') as f:
        f.write(model_json)

    model.save_weights(os.path.join(path,'model.h5'))

    print('Model Saved.')

# Save Load Model
def load_model(path):
    f = open(os.path.join(path,'model.json'), 'r')
    model_json = f.read()
    f.close()

    loaded_model = model_from_json(model_json)
    loaded_model.load_weights(os.path.join(path,'model.h5'))

    print("Model Loaded.")
    return loaded_model

# Parse *.midi file
def parse_midi(path):
    try:
        midi = Midi.PrettyMIDI(path)
        midi.remove_invalid_notes()
    except:
        midi = None
    return midi

# Filter Monophonic Instruments
def filter_mono(instruments, threshold=0.9):
    
    # Monophonic Percentage
    def percent(roll):
        one_hot = (roll.T > 0)
        notes = np.sum(one_hot, axis=1)
        per = np.count_nonzero(notes == 1) / np.count_nonzero(notes)
        return max(per, 0.0)
    
    return [x for x in instruments if percent(x.get_piano_roll()) >= threshold]


def create_note_sequences(midis, seq_len):
    
    def encode_seq(inst, seq_len):
        roll = inst.get_piano_roll(fs=4).T

        # Trim Beginning Silence
        a = np.sum(roll, axis=1)
        a = (a > 0).astype(float)
        roll = roll[np.argmax(a):]
    
        # One Hot Encode
        roll = (roll > 0).astype(float)
   
        # Add No-Event to Vocab
        a = a[np.argmax(a):]
        a = (a != 1).astype(float)
        roll = np.insert(roll, 0, a, axis=1)
        
        seqs = []
        for i in range(0, roll.shape[0] - seq_len - 1):
            seqs.append((roll[i:i + seq_len], roll[i + seq_len + 1]))
        return seqs
        
    X, y = [], []
    for f in midis:
        if f:
            mono_insts = filter_mono(f.instruments, THRESHOLD)
            for x in mono_insts:
                if len(x.notes) > seq_len:
                    note_seq = encode_seq(x, seq_len)
                    for _ in note_seq:
                        X.append(_[0])
                        y.append(_[1])
    X, y = np.asarray(X), np.asarray(y)
    return X, y


# Data Generator, shape: (batch_size, seq_len, vocab_size)
def data_gen(midi_files, batch_size=16, seq_len=32, num_threads=8, ram=150):
    
    # Check Multiprocessing
    if num_threads > 1:
        pool = ThreadPool(num_threads)
    
    cnt = 0
    l = len(midi_files)
    while 1:
        curr_files = midi_files[cnt: cnt + ram]
        cnt = (cnt + ram)%l
        
        if num_threads > 1:
            midis = pool.map(parse_midi, curr_files)
        else:
            midis = map(parse_midi, curr_files)
        
        data = create_note_sequences(midis, seq_len)
        i = 0
        while i < len(data[0]) - batch_size:
            X = data[0][i : i + batch_size]
            y = data[1][i : i + batch_size]
            yield (X,y)
            i += batch_size
        
        # Free Memory
        del midis, data
        

# Generate Music
def generate(model, prime, length):
    
    out = []
    prime = prime[random.randint(0, len(prime) - 1)]
    
    vocab_size = prime.shape[1]
    prime = prime.tolist()
    
    for i in range(length):
        pred = model.predict(np.expand_dims(np.asarray(prime), 0))

        # Sampling According to probability distribution
        idx = np.random.choice(range(0, vocab_size), p=pred[0])

        pred = np.zeros(vocab_size)
        pred[idx] = 1
        
        out.append(pred)
        prime.pop(0)
        prime.append(pred)
    
    return out
    

# Write to midi format and save
def write_midi(out, instrument='Acoustic Grand Piano', path=None):
     
    midi = Midi.PrettyMIDI()
    
    cur_note, start_note = None, None 
    time = 0.

    instrument_program = Midi.instrument_name_to_program(instrument)
    instrument = Midi.Instrument(program=instrument_program)
   
    for x in out:

        note = np.argmax(x) - 1
        
        # Note Changed or not
        if note != cur_note:
            
            if cur_note is not None:
                if cur_note >=0:
                
                    i = Midi.Note(velocity=127, pitch=min(max(int(note),0),127), start=start_note, end=time)
                    instrument.notes.append(i)

            cur_note, start_note = note, time
            
        time += 1. / 4

    midi.instruments.append(instrument)
    
    #if path:
    #    midi.write(path)
        
    return midi

