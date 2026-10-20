import unittest
import torch
import numpy as np

from src.deep_learning.eeg_conformer import EEGConformer
from src.speech_decoder import ImaginedSpeechDecoder, generate_synthetic_speech_eeg, SPEECH_ROI_CHANNELS

class TestSpeechDecoder(unittest.TestCase):
    
    def test_eeg_conformer_forward_shape(self):
        batch_size = 4
        n_channels = 15
        n_samples = 640
        n_classes = 5
        
        model = EEGConformer(n_channels=n_channels, n_samples=n_samples, n_classes=n_classes)
        x = torch.randn(batch_size, n_channels, n_samples)
        out = model(x)
        
        self.assertEqual(out.shape, (batch_size, n_classes))

    def test_synthetic_data_generator(self):
        X, y, classes = generate_synthetic_speech_eeg(n_trials_per_class=10, n_channels=15, n_samples=640)
        self.assertEqual(X.shape, (50, 15, 640))
        self.assertEqual(len(y), 50)
        self.assertEqual(len(classes), 5)

    def test_speech_decoder_train_and_predict(self):
        X, y, classes = generate_synthetic_speech_eeg(n_trials_per_class=8, n_channels=15, n_samples=640)
        decoder = ImaginedSpeechDecoder(n_channels=15, n_samples=640)
        
        history = decoder.train_on_data(X, y, epochs=2, batch_size=8)
        self.assertIn("val_acc", history)
        self.assertTrue(decoder.is_trained)
        
        res = decoder.predict(X[0])
        self.assertIn("predicted_word", res)
        self.assertIn(res["predicted_word"], classes)
        self.assertGreaterEqual(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)

if __name__ == '__main__':
    unittest.main()
