from collections import defaultdict
from copy import deepcopy
from jittor import optim
from typing import Dict, List, Optional
from tqdm import tqdm

import jittor as jt
import os

from ..data.asset import Asset
from ..data.dataset import PCDatasetModule
from ..model.spec import ModelSpec

def _get_item(x):
    if isinstance(x, jt.Var):
        return x.item()
    return x


def get_dist_info():
    if jt.in_mpi and jt.mpi is not None:
        return jt.mpi.world_rank(), jt.mpi.world_size()
    return 0, 1


def is_main_process():
    return get_dist_info()[0] == 0

def get_optimizer(optimizer_config, model):
    optimizer_config = deepcopy(optimizer_config)
    __target__ = optimizer_config.pop('__target__')
    MAPPING = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
    }
    if __target__ not in MAPPING:
        raise ValueError(f"unsupported optimizer: {__target__}")
    OptimizerClass = MAPPING[__target__]
    optimizer = OptimizerClass(model.parameters(), **optimizer_config)
    return optimizer

class DummyWriter():
    
    def __init__(self):
        pass
    
    def write(self, batch, prediction: List[Dict], dataset_module: Optional[PCDatasetModule]=None):
        pass

class DummySystem():
    
    def __init__(
        self,
        dataset_module: PCDatasetModule,
        model: ModelSpec,
        loss_config=None,
        optimizer_config=None,
        trainer_config=None,
        writer: Optional[DummyWriter]=None,
        
        ckpt_save_dir: str="experiments",
        ckpt_save_name: str="checkpoint",
    ):
        self.dataset_module = dataset_module
        self.model = model
        self.loss_config = loss_config
        self.ckpt_save_dir = ckpt_save_dir
        self.ckpt_save_name = ckpt_save_name
        self.writer = writer
        if trainer_config is None:
            trainer_config = {}
        self.epochs = trainer_config.get('epochs', 1)
        self.save_every = trainer_config.get('save_every', 1)
        self.save_every_steps = trainer_config.get('save_every_steps', 0)
        self.max_steps_per_epoch = trainer_config.get('max_steps_per_epoch', 0)
        self.lr_decay = trainer_config.get('lr_decay', 1.0)
        self.lr_decay_every = trainer_config.get('lr_decay_every', 1)
        self.loss_ema_alpha = trainer_config.get('loss_ema_alpha', 0.98)
        self.best_validation_loss = None
        
        if optimizer_config is not None and model is not None:
            self.optimizer = get_optimizer(optimizer_config, model)
        else:
            self.optimizer = None
        
        self._validation_loss = defaultdict(list)
    
    def forward(self, batch, validate: bool=False): # return loss sum
        loss_dict = self.model.training_step(batch)
        assert isinstance(loss_dict, dict), "loss_dict must be a dict containing loss/metrics"
        assert self.loss_config is not None, "do not have loss_confing"
        loss_sum = 0.
        if validate:
            assets: List[Asset] = [a for a in batch['asset']]
            cls = assets[0].cls # guaranteed to be the same cls in dataloader
            for name in loss_dict:
                assert name in self.loss_config, f'unspecified loss {name}'
                self._validation_loss[f"val/{cls}_{name}"].append(_get_item(loss_dict[name]))
                loss_sum += self.loss_config[name] * loss_dict[name]
            self._validation_loss[f"val/{cls}_loss_sum"].append(_get_item(loss_sum))
            # TODO: log
            # self.log('val/loss_sum', loss_sum, prog_bar=True, logger=True, sync_dist=True, batch_size=len(assets))
        else:
            for name in loss_dict:
                assert name in self.loss_config, f"unspecified loss name: `{name}`"
                if self.loss_config[name] > 0:
                    loss_sum += self.loss_config[name] * loss_dict[name]
            loss_dict['loss_sum'] = loss_sum
            # TODO: log
            # # add train prefix to loss_dict
            # prefixed_loss_dict = {f"train/{k}": v for k, v in loss_dict.items()}
            # d = dict(sorted(prefixed_loss_dict.items()))
        if not isinstance(loss_sum, jt.Var):
            return jt.array(loss_sum)
        return loss_sum
    
    def on_train_epoch_start(self):
        pass
    
    def on_train_batch_start(self):
        pass
    
    def training_step(self, batch):
        return self.forward(batch, validate=False)
    
    def on_train_batch_end(self):
        pass
    
    def on_train_epoch_end(self):
        pass
    
    def on_validation_epoch_start(self):
        self._validation_loss = defaultdict(list)
    
    def on_validation_batch_start(self):
        pass
    
    def validation_step(self, batch):
        assert self.loss_config is not None, "do not have loss_confing"
        return self.forward(batch, validate=True)
    
    def on_validation_batch_end(self):
        pass
    
    def on_validation_epoch_end(self):
        if not self._validation_loss:
            return
        summary = {}
        for key, values in self._validation_loss.items():
            if values:
                summary[key] = sum(values) / len(values)
        if summary:
            msg = ", ".join(f"{k}: {v:.6f}" for k, v in sorted(summary.items()))
            print(f"Validation summary: {msg}")
    
    def on_before_optimizer_step(self, optimizer):
        pass
    
    def on_predict_epoch_start(self):
        pass
    
    def on_predict_batch_start(self):
        pass
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        return self.model.predict_step(batch)
    
    def on_predict_batch_end(self):
        pass
    
    def on_predict_epoch_end(self):
        pass
    
    def train(self):
        assert self.optimizer is not None, "optimizer is None, cannot train"
        self.model.set_predict(False)
        for epoch in range(self.epochs):
            self.model.train()
            self.on_train_epoch_start()
            train_dataloader = self.dataset_module.train_dataloader()
            assert train_dataloader is not None, "train_dataloader is None"
            pbar = tqdm(
                train_dataloader,
                total=len(train_dataloader)//train_dataloader.batch_size,
                disable=not is_main_process(),
            ) # type: ignore
            epoch_loss_sum = 0.0
            epoch_loss_count = 0
            loss_ema = None
            for batch_idx, batch in enumerate(pbar):
                self.on_train_batch_start()
                loss = self.training_step(batch)
                loss_item = _get_item(loss)
                epoch_loss_sum += loss_item
                epoch_loss_count += 1
                if loss_ema is None:
                    loss_ema = loss_item
                else:
                    loss_ema = self.loss_ema_alpha * loss_ema + (1 - self.loss_ema_alpha) * loss_item
                self.optimizer.zero_grad()
                self.optimizer.backward(loss)
                if is_main_process():
                    pbar.set_description(f"Epoch {epoch}, Loss: {loss_item:.6f}, EMA: {loss_ema:.6f}")
                self.on_before_optimizer_step(self.optimizer)
                self.optimizer.step()
                if (
                    is_main_process()
                    and self.save_every_steps > 0
                    and (batch_idx + 1) % self.save_every_steps == 0
                ):
                    os.makedirs(self.ckpt_save_dir, exist_ok=True)
                    checkpoint_path = os.path.join(
                        self.ckpt_save_dir,
                        f'{self.ckpt_save_name}_e{epoch}_s{batch_idx + 1}.pkl',
                    )
                    self.model.save(checkpoint_path)
                    print(f"Saved step checkpoint: {checkpoint_path}")
                self.on_train_batch_end()
                if self.max_steps_per_epoch > 0 and (batch_idx + 1) >= self.max_steps_per_epoch:
                    break
            if is_main_process() and epoch_loss_count > 0:
                print(
                    f"Epoch {epoch} train mean loss: {epoch_loss_sum / epoch_loss_count:.6f}, "
                    f"ema loss: {loss_ema:.6f}"
                )
            self.on_train_epoch_end()
            
            self.model.eval()
            validate_dataloader = self.dataset_module.validate_dataloader()
            if validate_dataloader is not None:
                self.on_validation_epoch_start()
                if isinstance(validate_dataloader, dict):
                    for name, dataloader in validate_dataloader.items():
                        pbar = tqdm(
                            dataloader,
                            total=len(dataloader)//dataloader.batch_size,
                            disable=not is_main_process(),
                        )
                        for batch in pbar:
                            self.on_validation_batch_start()
                            loss = self.validation_step(batch)
                            if is_main_process():
                                pbar.set_description(f"Epoch {epoch}, Validate {name}, Loss: {_get_item(loss)}")
                            self.on_validation_batch_end()
                else:
                    pbar = tqdm(
                        validate_dataloader,
                        total=len(validate_dataloader)//validate_dataloader.batch_size,
                        disable=not is_main_process(),
                    )
                    for batch in pbar:
                        self.on_validation_batch_start()
                        loss = self.validation_step(batch)
                        if is_main_process():
                            pbar.set_description(f"Epoch {epoch}, Validate, Loss: {_get_item(loss)}")
                        self.on_validation_batch_end()
                self.on_validation_epoch_end()

            if is_main_process():
                os.makedirs(self.ckpt_save_dir, exist_ok=True)
                if self.save_every > 0 and (epoch + 1) % self.save_every == 0:
                    checkpoint_path = os.path.join(self.ckpt_save_dir, f'{self.ckpt_save_name}_{epoch}.pkl')
                    self.model.save(checkpoint_path)

            if is_main_process() and self._validation_loss:
                val_values = self._validation_loss.get('val/shapenet_loss_sum', [])
                if val_values:
                    val_loss = sum(val_values) / len(val_values)
                    if self.best_validation_loss is None or val_loss < self.best_validation_loss:
                        self.best_validation_loss = val_loss
                        best_path = os.path.join(self.ckpt_save_dir, f'{self.ckpt_save_name}_best.pkl')
                        self.model.save(best_path)

            if self.lr_decay != 1.0 and self.lr_decay_every > 0 and (epoch + 1) % self.lr_decay_every == 0:
                if hasattr(self.optimizer, 'lr'):
                    self.optimizer.lr *= self.lr_decay
                    if is_main_process():
                        print(f"Learning rate decayed to {self.optimizer.lr}")
    
    def predict(self):
        # only iterate once
        self.model.set_predict(True)
        self.model.eval()
        self.on_predict_epoch_start()
        predict_dataloader = self.dataset_module.predict_dataloader()
        assert predict_dataloader is not None, "predict_dataloader is None"
        if not isinstance(predict_dataloader, dict):
            predict_dataloader = {"predict": predict_dataloader}
        for dataloader_name, dataloader in predict_dataloader.items():
            pbar = tqdm(dataloader, total=len(dataloader)//dataloader.batch_size) # type: ignore
            for batch_idx, batch in enumerate(pbar):
                self.on_predict_batch_start()
                output = self.predict_step(batch, batch_idx)
                if self.writer is not None:
                    self.writer.write(batch, output, dataset_module=self.dataset_module)
                pbar.set_description(f"Predicting {dataloader_name}, Batch {batch_idx}")
