import os
import cv2
import librosa
import numpy as np
import torch.utils.data as data


class AudioDataset(data.Dataset):
    def __init__(
            self, df, conf, input_dir, imgs_dir,
            class_names, audio_transform, image_transform, subset=100):
        self.conf = conf
        self.audio_transform = audio_transform
        self.image_transform = image_transform

        if subset != 100:
            assert subset < 100
            # train and validate on subsets
            num_rows = df.shape[0]*subset//100
            df = df.iloc[:num_rows]

        files = df['{{ cookiecutter.file_column }}']
        assert isinstance(files[0], str), (
            f'column {df.columns[0]} must be of type str')
        self.files = [os.path.join(input_dir, imgs_dir, f) for f in files]

        labels = df['{{ cookiecutter.label_column }}']
        num_samples = len(files)
        num_classes = len(class_names)
        class_map = {class_names[i]: i for i in range(num_classes)}
        self.labels = np.zeros((num_samples, num_classes), dtype=np.float32)
        for i in range(num_samples):
            row_labels = [class_map[token] for token in labels[i].split(' ')]
            self.labels[i, row_labels] = 1.0

    def __getitem__(self, index):
        conf = self.conf
        filename = self.files[index]
        assert os.path.isfile(filename)

        num_samples =  conf.duration*conf.sample_rate
        sound, rate = librosa.load(
            filename, sr=conf.sample_rate, offset=conf.offset, duration=conf.duration)
        while (sound.shape[0] < num_samples):
            # pad to required length by duplicating data
            sound = np.hstack((sound, sound[:(num_samples - sound.shape[0])]))
        assert sound.shape[0] == num_samples

        if self.audio_transform:
            sound = self.audio_transform(samples=sound, sample_rate=conf.sample_rate)
        assert sound.shape[0] == num_samples
        spec = librosa.feature.melspectrogram(
            y=sound, sr=conf.sample_rate, n_fft=conf.num_fft, hop_length=conf.hop_length,
            n_mels=conf.num_mels,  fmin=conf.min_freq, fmax=conf.max_freq)
        img = librosa.power_to_db(spec, ref=np.max)
        img -= img.min()
        img /= img.max()
        img *= 255
        img = img.round().astype(np.uint8)
        img = cv2.resize(
            img, (conf.spectrogram_width, conf.num_mels),
            interpolation=cv2.INTER_AREA)
        img = np.stack((img, img, img), axis=2)
        if self.image_transform:
            img = self.image_transform(image=img)['image']
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.files)
