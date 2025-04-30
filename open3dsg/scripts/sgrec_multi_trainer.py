import argparse
import os
import json
import numpy as np
import pickle
import open3d as o3d
from torch.utils.tensorboard import SummaryWriter
from open3dsg.config.config import CONF
from open3dsg.models.SGRec3D import SGRec3D
from open3dsg.scripts.sgrec_eval import SGRecEvaluator
import os
import torch
from itertools import accumulate
from tqdm import tqdm
torch.manual_seed(1124)
import torch.multiprocessing as mp
# Add at the beginning of your script
mp.set_start_method('spawn', force=True)
from collections import defaultdict
from open3dsg.data.sgrec_dataloader import SGRecDataset, DataLoaderX

def finetune_sgrec3d(model, dataloader, optimizer, num_epochs, save_dir, log_dir, save_interval, output_dir, ckpt_dir, scheduler, config, evaulator):
    log_dir = os.path.join(log_dir, 'finetune_tensorboard_logs')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    # load checkpoint
    ckpt = torch.load(ckpt_dir)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    # if "optimizer_state_dict" in ckpt:
    #     optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        
    epoch = 0
    global_step = 0
    
    model.train()
    for epoch in tqdm(range(epoch, num_epochs)):
        cum_recon_loss = defaultdict(float)
        loss_keys = {}
        for batch_idx, batch in enumerate(dataloader):
            model.train()
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
                elif key == 'triples':
                    batch[key] = [t.to(device) for t in batch[key]]
                        
            global_step = epoch * len(dataloader) + batch_idx
            
            output_dict = model.scene_graph_encoder_forward(batch)
            sg_loss, sg_loss_dict = model.sg_logits_loss(output_dict, grad_accum_steps=GRADIENT_ACCUMULATION_STEPS)
            
            if not loss_keys:
                loss_keys = sg_loss_dict.keys()
            for loss_key in sg_loss_dict:
                cum_recon_loss[loss_key] += sg_loss_dict[loss_key] * config['batch_size']
            # backward pass
            sg_loss.backward()
            # weights update
            if ((batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0) or (batch_idx + 1 == len(dataloader)):
                optimizer.step()
                optimizer.zero_grad()
            
            if global_step % 5000 == 0:
                # # save output_dict to a file
                # output_dirr = os.path.join(output_dir, f"output_{global_step}")
                # os.makedirs(output_dirr, exist_ok=True)
                # torch.save(output_dict, os.path.join(output_dirr, f"training_batch_{global_step}.pth"))
                
                file_name = f"finetuned_epoch_{global_step}.pth"
                ckpt_dir = os.path.join(save_dir, file_name)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, ckpt_dir)
                print(f"Saved checkpoint to {ckpt_dir}")
                
                eval_idx = [5, 10, 100, 675, 1000, 1650, 1666, 2037, 2131, 2572, 3449]
                if evaluator is not None:
                    for idx in eval_idx:
                        evaluator.eval_one_scene(idx, prefix=str(global_step))
                        metrics = evaluator.eval_metrics(idx)
                        for key in metrics:
                            writer.add_scalar(f"metrics_{idx}/{key}", metrics[key], global_step)
                
            if batch_idx % 100 == 0:
                for loss_key in sg_loss_dict:
                    writer.add_scalar('info/' + loss_key, sg_loss_dict[loss_key], global_step)
            
            torch.cuda.empty_cache()
            
        for loss_key in loss_keys:
            cum_recon_loss[loss_key] /= len(dataloader)
            writer.add_scalar('training/' + loss_key, cum_recon_loss[loss_key], epoch)
        
        tqdm.write(f"Epoch {epoch+1} - finetune Loss: {cum_recon_loss['sg_logits_loss']:.4f}")
        
        if scheduler is not None:
            scheduler.step()
            writer.add_scalar('training/lr', scheduler.get_last_lr()[0], epoch)
    # Close the TensorBoard writer
    writer.close()

def pretrain_sgrec3d(model, dataloader, optimizer, num_epochs, save_dir, log_dir, save_interval, output_dir, load_ckpt=None, scheduler=None, config=None, eval_model=None):
    # Create TensorBoard writer
    log_dir = os.path.join(log_dir, 'pretrain_tensorboard_logs')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Initialize the GradScaler for AMP
    # scaler = amp.GradScaler(enabled=use_amp)
    
    if load_ckpt is not None:
        print(f"Loading checkpoint from {load_ckpt}")
        ckpt = torch.load(load_ckpt)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
            
        if "optimizer_state_dict" in load_ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        resume_epoch = int(load_ckpt.split('_')[-1].split('.')[0])
        resume_epoch = resume_epoch // 1000 if resume_epoch > 1000 else resume_epoch
        print(f"Resuming training from epoch {resume_epoch}")
        
        if model.use_pretrained_atlasnet:
            model.load_pretrained_atlasnet()
            print("Loaded pretrained AtlasNet weights again because use_pretrained_atlasnet is True")
    else:
        resume_epoch = 0
    model.train()
    # print(f"Model is on device: {model.device}")
    
    eval_idx = [5, 10, 100, 675, 1000, 1650, 1666, 2037, 2131, 2572, 3449]

        
    global_step = resume_epoch * len(dataloader)
    
    if eval_model is not None:
        for idx in eval_idx:
            eval_model.eval_one_scene(idx, prefix="init")
            metrics = eval_model.eval_metrics(idx)
            for key in metrics:
                writer.add_scalar(f"metrics_{idx}/{key}", metrics[key], global_step)
    
    for epoch in tqdm(range(resume_epoch, num_epochs)):
        cum_recon_loss = defaultdict(float)
        loss_keys = {}
        for batch_idx, batch in enumerate(dataloader):
            model.train()
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
                elif key == 'triples':
                    batch[key] = [t.to(device) for t in batch[key]]
                        
            global_step = epoch * len(dataloader) + batch_idx
            
            output_dict = model(batch)
            
            recon_loss, recon_loss_dict = model.reconstruction_loss(output_dict, grad_accum_steps=GRADIENT_ACCUMULATION_STEPS)
            # print("Reconstruction loss: ", recon_loss)
            # print("Reconstruction loss dict: ", recon_loss_dict)
            
            if not loss_keys:
                loss_keys = recon_loss_dict.keys()
            for loss_key in recon_loss_dict:
                cum_recon_loss[loss_key] += recon_loss_dict[loss_key] * config['batch_size']
    
            # backward pass
            recon_loss.backward()

            # weights update
            if ((batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0) or (batch_idx + 1 == len(dataloader)):
                optimizer.step()
                optimizer.zero_grad()


            # Save the decoded shape every 5000 global steps
            if global_step % 5000 == 0:
                # save output_dict to a file
                output_dirr = os.path.join(output_dir, f"output_{global_step}")
                os.makedirs(output_dirr, exist_ok=True)
                torch.save(output_dict, os.path.join(output_dirr, f"training_batch_{global_step}.pth"))
                
                file_name = f"pretrained_epoch_{global_step}.pth"
                ckpt_dir = os.path.join(save_dir, file_name)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, ckpt_dir)
                print(f"Saved checkpoint to {ckpt_dir}")
                
                swap_yz = config['swap_yz'] if 'swap_yz' in config else False
                re_angle = config['re_angle'] if 're_angle' in config else False
                evalulator = SGRecEvaluator(config=config,
                                            model_path=ckpt_dir,
                                            dataset=sgrec_dataset,
                                            device=device,
                                            ply_saving_path=output_dir,
                                            swap_yz=swap_yz,
                                            re_angle=re_angle)
                
                for idx in eval_idx:
                    evalulator.eval_one_scene(idx, prefix=str(global_step))
                    metrics = evalulator.eval_metrics(idx)
                    for key in metrics:
                        writer.add_scalar(f"metrics_{idx}/{key}", metrics[key], global_step)
                    
                print(f"Evaluation complete with Evaluator config swap_yz: {swap_yz}, re_angle: {re_angle}")
                evaluator = eval_model
                for idx in eval_idx:
                    evaluator.eval_one_scene(idx, prefix=str(global_step)+"_model_in_training")
                    metrics = evalulator.eval_metrics(idx)
                    for key in metrics:
                        writer.add_scalar(f"metrics_model_{idx}/{key}", metrics[key], global_step)
                
            if batch_idx % 100 == 0:
                for loss_key in recon_loss_dict:
                    writer.add_scalar('info/' + loss_key, recon_loss_dict[loss_key], global_step)

              
            # del batch
            # del output_dict
            # del recon_loss
            # del recon_loss_dict
            torch.cuda.empty_cache()
        
        for loss_key in loss_keys:
            cum_recon_loss[loss_key] /= len(dataloader)
            writer.add_scalar('training/' + loss_key, cum_recon_loss[loss_key], epoch)
        
        tqdm.write(f"Epoch {epoch+1} - Reconstruction Loss: {cum_recon_loss['total_loss']:.4f}")
        
        if scheduler is not None:
            scheduler.step()
            writer.add_scalar('training/lr', scheduler.get_last_lr()[0], epoch)
    # Close the TensorBoard writer
    writer.close()
    
def load_sgrec_config(config_path="/home/ubuntu/SceneGraphEdit/open3dsg/config/default_config.json"):
    # check if config_path exists
    if not os.path.exists(config_path):
        print(f"Config file {config_path} does not exist. Try searching in the default config path.")
        config_path = "/home/ubuntu/SceneGraphEdit/open3dsg/config/" + config_path
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} does not exist.")
    try:
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
            config = loaded_config
        print(f"Loaded configuration from {config_path}")
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
    
    return config

def print_gpu_memory_stats(message=""):
    """Print current GPU memory usage with an optional message."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"{message} GPU memory: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved")

if __name__ == "__main__":
    config_path = "/home/ubuntu/SceneGraphEdit/open3dsg/config/config_v20_ft.json"
    
    config = load_sgrec_config(config_path=config_path)
        
    print("Using Configuration:")
    print("-"*265)
    print(config)
    print("-"*265)
    use_data_parallel = config['use_data_parallel']
    device = config['device']
    num_epochs = config['num_epochs']
    save_interval = config['save_interval']
    save_dir = config['save_dir']
    output_dir = config['output_dir']
    use_amp = config.get('use_amp', False)
    if not os.path.exists(save_dir):
        print(f"Creating save directory {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
    if not os.path.exists(output_dir):
        print(f"Creating output directory {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(config['log_dir']):
        print(f"Creating log directory {config['log_dir']}")
        os.makedirs(config['log_dir'], exist_ok=True) 
    
    GRADIENT_ACCUMULATION_STEPS = config['gradient_accumulation_steps']
    hparams = config['hparams']
    print("Re-angle:", config['re_angle'])
    print("Downsample points:", config['downsample_points'])
    sgrec_dataset = SGRecDataset(max_objects=hparams['max_nodes'],
                                 max_rels=hparams['max_edges'],
                                 num_obj_classes=hparams['num_obj_classes'],
                                 num_pred_classes=hparams['num_rel_classes'],
                                 data_root_dir=config['data_dir'],
                                 hashes=config['data_hash_dir'],
                                 device=device,
                                 downsample_points=config['downsample_points'],
                                 re_angle=config['re_angle'],)
    dataloader = DataLoaderX(sgrec_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=sgrec_dataset.collate_fn, num_workers=8, pin_memory=True)
    print(f"Dataloader length: {len(dataloader)}")
    model = SGRec3D(hparams, device=device)
    # evaluator = SGRecEvaluator(config=config, model=model, dataset=sgrec_dataset, device=device, ply_saving_path=output_dir)
    
    
    # evaluator = SGRecEvaluator(model_path=model_path,
    #                            dataset_path=dataset_path,
    #                            config={'hparams':hparams},
    #                            device='cuda:0',
    #                            swap_yz=False)
    
    # model_path = "/home/ubuntu/SceneGraphEdit/sgrec_pretrained_checkpoints/v7/pretrained_epoch_200.pth"
    # evaluator = SGRecEvaluator(config=config, model_path=model_path, dataset=sgrec_dataset, device=device, ply_saving_path=output_dir)
    evaluator = SGRecEvaluator(config=config,
                    model=model,
                    dataset=sgrec_dataset,
                    device=device,
                    ply_saving_path=output_dir,
                    re_angle=config['re_angle'] if 're_angle' in config else False,
                    swap_yz=config['swap_yz'] if 'swap_yz' in config else False)
    
    print("Model initialized")
    # print number of parameters, vram usage
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))
    print("VRAM usage:", sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024, "MB")
            
    if hparams['pretraining_mode']:
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)
        tmx = num_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmx, eta_min=1e-5)
        pretrain_sgrec3d(
            model=model,
            dataloader=dataloader, 
            optimizer=optimizer,
            num_epochs=num_epochs,
            save_dir=save_dir,
            log_dir=config['log_dir'],
            save_interval=save_interval,
            output_dir=output_dir,
            load_ckpt=config.get('load_ckpt', None),
            scheduler=scheduler,
            config=config,
            eval_model=evaluator
        )
    else:
        # Load the pretrained model
        load_ckpt = config.get('load_ckpt', None)
        if load_ckpt is None:
            raise ValueError("Pretraining mode is False, but no checkpoint provided for loading.")
        
        print(f"Loading checkpoint from {load_ckpt}")
        print("RUNNING IN FINE TUNE MODE !!!!!!!!!!!!!!!!!!")
            
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
        
        evaluator = SGRecEvaluator(config=config,
            model=model,
            dataset=sgrec_dataset,
            device=device,
            ply_saving_path=output_dir,
            re_angle=config['re_angle'] if 're_angle' in config else False,
            swap_yz=config['swap_yz'] if 'swap_yz' in config else False)
        
        finetune_sgrec3d(
            model=model,
            dataloader=dataloader, 
            optimizer=optimizer,
            num_epochs=num_epochs,
            save_dir=save_dir,
            log_dir=config['log_dir'],
            save_interval=save_interval,
            output_dir=output_dir,
            ckpt_dir=load_ckpt,
            scheduler=scheduler,
            config=config,
            evaulator=evaluator
        )