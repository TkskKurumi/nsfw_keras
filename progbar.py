import time
import shutil
class progbar:
    def __init__(self):
        self.last_title=None
        self.start_time=dict()
        self.i=0
    def finish(self,extra=''):
        tm=time.time()
        tm_consumed=tm-self.start_time[self.last_title]
        print(self.last_title,'finished in %.1f seconds%s'%(tm_consumed,extra))
    def __call__(self,title,i,tot):
        if(self.last_title is None):
            self.last_title=title
            
        if(title != self.last_title):
            tm=time.time()
            tm_consumed=tm-self.start_time[self.last_title]
            print(self.last_title,'finished in %.1f seconds'%tm_consumed)
            
        if(title not in self.start_time):
            self.start_time[title]=time.time()
        self.i+=1
        if(self.i&0b1111):
            return
        tm=time.time()
        tm_consumed=tm-self.start_time[title]
        
        progress=i/tot
        time_remain=tm_consumed*(tot-i)/i
        left="%s %.1f%% %.1f seconds remain:"%(title,progress*100,time_remain)
        right=''
        
        width=shutil.get_terminal_size().columns-len(left)-len(right)-10
        
        width_progress=int(width*progress)
        width_remain=width-width_progress
        if(width_progress>=0 and width_remain>=0):
            mid='['+'#'*width_progress+'.'*width_remain+']'     #[#####..]
        else:
            mid=''
        print(left,mid,right,end='\r',sep='')