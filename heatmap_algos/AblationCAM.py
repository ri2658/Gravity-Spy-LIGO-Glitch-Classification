import torch
import torch.nn.functional as F


class AblationCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.ablation_channel = None
        self._hook = None
        self._register_hook()

    def _register_hook(self):
        def forward_hook(module, inputs, output):
            self.activations = output.detach()

            if self.ablation_channel is None:
                return output

            ablated_output = output.clone()
            ablated_output[:, self.ablation_channel, :, :] = 0
            return ablated_output

        self._hook = self.target_layer.register_forward_hook(forward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        input_tensor = input_tensor.detach()

        with torch.no_grad():
            output = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())
        else:
            class_idx = int(class_idx)

        original_score = output[0, class_idx]

        self.ablation_channel = None
        with torch.no_grad():
            self.model(input_tensor)

        if self.activations is None:
            raise RuntimeError("Target layer activations were not captured.")

        activations = self.activations
        channel_count = activations.size(1)
        weights = []

        for channel_idx in range(channel_count):
            self.ablation_channel = channel_idx
            with torch.no_grad():
                ablated_output = self.model(input_tensor)
                ablated_score = ablated_output[0, class_idx]

            importance = (original_score - ablated_score).clamp(min=0.0)
            weights.append(importance)

        self.ablation_channel = None
        weights = torch.stack(weights, dim=0).view(1, channel_count, 1, 1)

        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        return cam[0, 0].cpu().numpy()
