from torch import nn


class ConvEncoder(nn.Module):

    def __init__(self, out_features=64):
        super(ConvEncoder, self).__init__()
        self.out_features = out_features
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2),  # b, 8, 29, 29
            nn.LeakyReLU(),
            nn.Conv2d(8, 1, 1)  # b, 1, 29, 29
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=1*29*29, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=self.out_features),
            nn.LeakyReLU()
        )

    def forward(self, image):
        batch_size = image.size(0)
        x = self.conv(image)
        return self.fc(x.view(batch_size, -1))


class ConvDecoder(nn.Module):

    def __init__(self, in_features=64):
        super(ConvDecoder, self).__init__()
        self.in_features = in_features
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=1*29*29),
            nn.LeakyReLU()
        )
        self.conv_transp = nn.Sequential(
            nn.ConvTranspose2d(1, 8, 3, stride=2),  # b, 5, 59, 59
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 1, 2),  # b, 5, 60, 60
            nn.Tanh()
        )

    def forward(self, x):
        batch_size = x.size(0)
        decoded = self.fc(x).view(batch_size, 1, 29, 29)
        return self.conv_transp(decoded)


class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, code_size=64):
        super(ConvolutionalAutoencoder, self).__init__()
        self.code_size = code_size
        self.encoder = ConvEncoder(out_features=code_size)
        self.decoder = ConvDecoder(in_features=code_size)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded