import _io
from glob import glob
from os import path
import shutil,time,os,traceback
last_prog=time.time()
last_title=""
i_prog=0
first_prog=dict()
def progbar(title,prog,width=20,print_finish=False):
	global i_prog,last_prog,last_title
	if(print_finish):
		t=time.time()
		print("%s finished in %.1f seconds  "%(last_title,t-first_prog[last_title]))
	i_prog+=1
	if(i_prog&0b11):
		return
	t=time.time()
	if((title!=last_title) and (last_title in first_prog)):
		print("%s finished in %.1f seconds  "%(last_title,t-first_prog[last_title]))
	last_title=title
	if(title not in first_prog):
		first_prog[title]=t
	if(t-last_prog>0.1):
		enmiao="#"*int(prog*width)
		enmiao+="."*max(width-len(enmiao),0)
		remain=(t-first_prog[title])/(prog+1e-10)*(1-prog)
		print(title,"["+enmiao+"] %.1f secs remain"%(remain),end='\r')
		last_prog=t

def shashi(s,encoding='utf-8',length=64):
	bytes=s.encode(encoding)
	ret=0
	for i in bytes:
		ret=(ret<<7)|int(i)
	mask=(1<<length)-1
	while(ret>>length):
		ret=(ret&mask)^(ret>>length)
	return ret
def readerhashi(f,length=64):
	ret=0
	f.seek(0)
	bytes=f.read()
	mask=(1<<length)-1
	for byte in bytes:#while(byte):
		ret=(ret<<7)|byte
		ret=(ret>>length)^(ret&mask)
		byte=f.read(1)
	return ret
def hashi(s,*args,**kwargs):
	if(isinstance(s,str)):
		return shashi(s,*args,**kwargs)
	elif(isinstance(s,int)):
		return s
	elif(isinstance(s,_io.BufferedReader)):
		return readerhashi(s,*args,**kwargs)
	else:
		return TypeError(type(s))
def base32(content,length=8):
	if(not isinstance(content,int)):
		return base32(hashi(content,length=length*5),length=length)
	ch='0123456789abcdefghijklmnopqrstuvwxyz'
	ret=[]
	
	mask=(1<<(length*5))-1
	while(content>>(length*5)):
		content=(content>>(length*5))^(content&mask)
	mask=0b11111

	for i in range(length):
		ret.append(ch[content&mask])
		content>>=5
	return ''.join(ret[::-1])


def merge(src,dst,rm=True):
	global last_prog,last_title,i_prog,first_prog
	last_prog=time.time()
	last_title=""
	i_prog=0
	first_prog=dict()

	name2src=dict()
	ls=list(glob(path.join(src,'*')))
	length=len(ls)
	for idx,i in enumerate(ls):
		progbar('copying',idx/length)
		f=open(i,'rb')
		name=base32(f,length=8)
		f.close()
		ext=path.splitext(i)[-1]
		
		newname=path.join(dst,name+ext)
		if(path.exists(newname)):
			print('\n',i,'exists',newname,end='\r')
		else:
			shutil.copy(i,newname)
			print('\n',i,'copied',newname,end='\r')
		if(rm):
			try:
				os.remove(i)
			except Exception:
				traceback.print_exc()
		
merge(src=r'M:\Weiyun Sync\code\nsfw_keras\datas\to_pos',
dst=r'M:\Weiyun Sync\code\nsfw_keras\datas\pos')

merge(src=r'M:\Weiyun Sync\code\nsfw_keras\datas\to_neg',
dst=r'M:\Weiyun Sync\code\nsfw_keras\datas\neg')

