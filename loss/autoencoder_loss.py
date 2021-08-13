from torch import nn


class AutoEncoderLoss(nn.Module):
    def __init__(self):
        super(AutoEncoderLoss, self).__init__()

        self.encoder_criterion = nn.MSELoss()
        self.decoder_criterion = nn.BCELoss()

    def forward(self, modified_audio, original_audio, reconstructed_message, original_message):

        encoder_loss = self.encoder_criterion(modified_audio, original_audio)
        decoder_loss = self.decoder_criterion(reconstructed_message, original_message)

        return encoder_loss + decoder_loss, encoder_loss, decoder_loss
