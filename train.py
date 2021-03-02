import argparse

import comet_ml
import torch
import kornia

from utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--api-key", required=True)

parser.add_argument("--model-name", default="transgan")
parser.add_argument("--model-size", default="s", help="s, m, l, xl")
parser.add_argument("--size", default=32, type=int)
parser.add_argument("--dataset", default="c10")
parser.add_argument("--g-lr", default=5e-5, type=float)
parser.add_argument("--d-lr", default=4e-4, type=float)
parser.add_argument("--criterion", default="ns", help="ns, mse")
parser.add_argument("--optimizer", default="adam")
# parser.add_argument("--g-batch-size", default=32, type=int)
# parser.add_argument("--d-batch-size", default=16, type=int)
parser.add_argument("--batch-size", default=64, type=int)
parser.add_argument("--beta1", default=0., type=float)
parser.add_argument("--beta2", default=0.99, type=float)
parser.add_argument("--patch", default=8, type=int)
parser.add_argument("--max-steps", default=20000, type=int)
parser.add_argument("--log-steps", default=1000, type=int, help="Log images every this step.")
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--diffaugment", action="store_true")
args = parser.parse_args()
args.experiment_name = get_experiment_name(args)
if args.dry_run:
    args.max_steps = 1
    args.batch_size = 16

EMB = {
    "s":384,
    "m":512,
    "l":768,
    "xl":1024
}

scaler = torch.cuda.amp.GradScaler()
class Trainer:
    def __init__(self, logger, args):
        self.logger = logger
        self.args = args
        self.logger.log_parameters(vars(self.args))
        self.emb_dim = EMB[self.args.model_size]
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.curr_step = 1
        if self.args.diffaugment:
            self.transform = self._diffaugment()
        
    def fit(self, g, d, trian_dl):
        g, d = g.to(self.device), d.to(self.device)
        criterion = get_criterion(self.args)
        optimizers = get_optimizer(g, d, self.args)
        while self.curr_step <= self.args.max_steps:
            for img, _ in train_dl:
                img = img.to(self.device)
                self.train_step(g, d, img, criterion, optimizers)
                if self.curr_step % self.args.log_steps==0 or self.args.dry_run:
                    g.eval()
                    z = torch.randn(32, self.emb_dim, device=self.device)
                    with torch.cuda.amp.autocast():
                        with torch.no_grad():
                            f_img = g(z) 
                    self._log_image(f_img)
                if self.args.dry_run:
                    break
        self._log_weight(g)

    def train_step(self, g, d, img, criterion, optimizers):
        g.train()
        d.train()
        g_opt, d_opt = optimizers
        g_opt.zero_grad()
        d_opt.zero_grad()
        real = torch.ones(img.size(0), 1, device=self.device)
        fake = torch.ones(img.size(0), 1, device=self.device) * 0.1
        z = torch.randn(img.size(0), self.emb_dim, device=self.device)

        # D
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                f_img = g(z)
                if self.args.diffaugment:
                    img = self.transform(img)
                    self._log_image(img)
                    f_img = self.transform(f_img)
            out = d(img)
            f_out = d(f_img)
            loss = (criterion(out, real*0.9) + criterion(f_out, fake))/2.
        scaler.scale(loss).backward()
        scaler.step(d_opt)
        scaler.update()
        self.logger.log_metric("d_loss", loss)

        # G
        with torch.cuda.amp.autocast():
            f_img = g(z)
            if self.args.diffaugment:
                f_img = self.transform(f_img)
            f_out = d(f_img)
            loss = criterion(f_out, real)
        scaler.scale(loss).backward()
        scaler.step(g_opt)
        scaler.update()
        self.logger.log_metric("g_loss", loss)

        self.curr_step += 1
    
    def _log_image(self, img):
        grid = torchvision.utils.make_grid(img.detach().cpu(), nrow=4)
        self.logger.log_image(grid.permute(1,2,0), step=self.curr_step)

    def _log_weight(self, g):
        g.eval()
        filename=f"weights/{self.args.experiment_name}.pth"
        torch.save(g.cpu().state_dict(), filename)
        self.logger.log_asset(filename, file_name=filename)
    
    def _diffaugment(self):
        transform = nn.Sequential(
            kornia.augmentation.RandomCrop((self.args.size, self.args.size), padding=self.args.size//8),
            kornia.augmentation.RandomErasing(scale=(0.25,0.25+1e-8), ratio=(1.,1.), p=0.3),# Cutout
            kornia.augmentation.ColorJitter(brightness=0.5, contrast=[0.5, 1.5],saturation=[0, 2])
        )
        return transform


if __name__ == "__main__":
    logger = comet_ml.Experiment(
        api_key=args.api_key,
        project_name="image-generation-pytorch",
        auto_metric_logging=False
    )
    logger.set_name(args.experiment_name)
    train_dl = get_dataloader(args)
    g,d = get_model(args)
    trainer = Trainer(logger=logger, args=args)
    trainer.fit(g, d, train_dl)


