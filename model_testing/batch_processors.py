import torch


class LotterErrorProcessor:

    def __init__(self, buffering_frames=10):
        self.buffering_frames = buffering_frames

    def __call__(self, model, data):
        batch_size   = data.size(0)
        sequence_len = data.size(1)

        model_output, target = [], []
        for i in range(self.buffering_frames, sequence_len - 1):
            state = model.init_hidden(batch_size)
            for t in range(i - self.buffering_frames, i):
                output, state = model(data[:,t,:], state)

            model_output.append(output)
            target.append(data[:,i,:])

        return torch.stack(model_output, dim=1), torch.stack(target, dim=1)


class LotterErrorProcessorAuto(LotterErrorProcessor):

    def __call__(self, model, data):
        model.decoding_steps = 1
        sequence_len = data.size(1)

        model_output, target = [], []
        for i in range(self.buffering_frames, sequence_len - 1):
            data_begin = i - self.buffering_frames
            input_data = data[:, data_begin:i, :]
            prediction = model(input_data)

            model_output.append(prediction)
            target.append(data[:,i,:])

        return torch.cat(model_output, dim=1), torch.stack(target, dim=1)


class LotterErrorProcessorAutoMultiDec(LotterErrorProcessor):

    def __call__(self, model, data):
        model.decoding_steps = 2
        sequence_len = data.size(1)

        model_output, target = [], []
        for i in range(self.buffering_frames, sequence_len - 2):
            data_begin = i - self.buffering_frames
            input_data = data[:, data_begin:i, :]
            _, prediction = model(input_data)

            model_output.append(prediction[:,1,:].unsqueeze(1))
            target.append(data[:,i+1,:])

        return torch.cat(model_output, dim=1), torch.stack(target, dim=1)


class LongTermProcessor:

    def __init__(self, prediction_length, buffering_frames=10):
        self.pred_len = prediction_length
        self.buffering_frames = buffering_frames

    def __call__(self, model, data):
        batch_size   = data.size(0)
        sequence_len = data.size(1)

        model_output, target = [], []
        state = model.init_hidden(batch_size)
        for t in range(self.buffering_frames + self.pred_len): 
            if t < self.buffering_frames:
                output, state = model(data[:,t,:,:,:], state)
            else:
                output, state = model(output, state)
                if t < sequence_len - 1:
                    ground_truth = data[:,t+1,:,:,:]

                model_output.append(output)
                target.append(ground_truth)

        return torch.stack(model_output, dim=1), torch.stack(target, dim=1)


class LongTermProcessorAuto(LongTermProcessor):

    def __call__(self, model, data):
        model.decoding_steps = self.pred_len
        batch_size   = data.size(0)
        sequence_len = data.size(1)

        prediction = model(data[:,:self.buffering_frames,:])

        gt_end = min(sequence_len, self.buffering_frames + self.pred_len)
        ground_truth = data[:, self.buffering_frames:gt_end, :]
        gt_len = ground_truth.size(1)
        if gt_len < self.pred_len:
            missing_frames = self.pred_len - gt_len
            padding = ground_truth[:, -1, :].unsqueeze(1).expand(batch_size, missing_frames, -1, -1, -1)
            ground_truth = torch.cat([ground_truth, padding], dim=1)

        return prediction, ground_truth


class LongTermProcessorAutoMultiDec(LongTermProcessor):

    def __call__(self, model, data):
        model.decoding_steps = self.pred_len
        batch_size   = data.size(0)
        sequence_len = data.size(1)

        _, prediction = model(data[:,:self.buffering_frames,:])

        gt_end = min(sequence_len, self.buffering_frames + self.pred_len)
        ground_truth = data[:, self.buffering_frames:gt_end, :]
        if gt_end > sequence_len:
            frame_size = data.size()[2:]
            missing_frames = gt_end - sequence_len
            padding = ground_truth.new_zeros((batch_size, missing_frames) + frame_size)
            ground_truth = torch.cat([ground_truth, padding], dim=1)

        return prediction, ground_truth
