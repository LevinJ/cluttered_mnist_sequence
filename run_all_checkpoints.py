import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import os
import re
import argparse
import glob
import os


from snrglobaldef import g_modellogdir

class RunAllCheckpoints(object):
    def __init__(self):
        
        
        
        return
    
    def __get_all_ckpt(self, checkpoint_path):
        res = []
        ckpt_files = glob.glob("{}/model.ckpt-*.index".format(g_modellogdir))
        for file_name in ckpt_files:
            file_id = os.path.basename(file_name)[11:-6]
            res.append(int(file_id))
        res.sort()
        return res
    
    def get_all_checkpoints(self,checkpoint_path):
#         
#         with open("{}/checkpoint".format(g_modellogdir)) as f:
#             content = f.readlines()
#         content = [x.strip() for x in content] 
#         checkpoints = []
#         for line in content:
#             m = re.search('all_model_checkpoint_paths:(.*)model.ckpt-(.*)"', line)
#             if m:
#                 num = m.group(2)
#                 checkpoints.append(num)
        
        checkpoints = self.__get_all_ckpt(checkpoint_path)
        min_step = self.min_step
        step = 100
        last_step = min_step
        sel_checkpoints = []
        for checkpoint in checkpoints:
            checkpoint = int(checkpoint)
            if checkpoint < min_step:
                continue
            if checkpoint == int(checkpoints[-1]):
                #the last checkpoint always get selected
                sel_checkpoints.append(checkpoint)
                continue
            if checkpoint >= last_step:
                sel_checkpoints.append(checkpoint)
                last_step = last_step + step
        if self.check_only_latest:
            #if we only want to evluate the latest checkpoints
            sel_checkpoints = [sel_checkpoints[-1]]
        return sel_checkpoints
    def parse_param(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-l', '--latest',  help='evaluate only the latest checkpoints',  action='store_true')
        parser.add_argument('-c', '--checkpoint_path',  help='which checkpoint(directory) to use',  default="")
        parser.add_argument('-m', '--min_step',  help='min_step of checkpoint to start with',  type=int, default=0)
        args = parser.parse_args()
        
        self.checkpoint_path = args.checkpoint_path
        self.check_only_latest = args.latest
        self.min_step = args.min_step
        
        if self.checkpoint_path != '':
            global g_modellogdir
            g_modellogdir = "{}/{}".format(g_modellogdir, self.checkpoint_path)
            
        return
    def run_all_checkpoints(self):
        
        sel_checkpoints = self.get_all_checkpoints(self.checkpoint_path)
        #for tine tuning checkpoint path, we can skip the first chckpoint since it's already calcuated 
#         if self.checkpoint_path != '' and (not self.check_only_latest):
#             sel_checkpoints = sel_checkpoints[1:]
                
        
        for checkpoint in sel_checkpoints:
            for split_name in ["train", "eval"]:
                
                checkpoint_file ="{}/model.ckpt-{}".format(g_modellogdir,  checkpoint)
#                 print("checkpoint {}, {} data".format(checkpoint_file, split_name))
                
                cmd_str = "python ./eval_model.py "
                
                cmd_str = '{} -s "{}" -c "{}"'.format(cmd_str, split_name, checkpoint_file)
                os.system(cmd_str)
            
        return
    
    
    def run(self):
        self.parse_param()
        self.run_all_checkpoints()
        
        
        
        return
    
    


if __name__ == "__main__":   
    obj= RunAllCheckpoints()
    obj.run()