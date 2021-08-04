"""
Change .avi movie files and convert them into mp4.
"""
import os
import glob

avilist=glob.glob('*.avi') # or mp4
avilist.sort()

for avifile in avilist:
    mp4file=avifile[:-3] + 'mp4'
    # os.system('ffmpeg -y -i %s -pix_fmt yuv420p %s'%(avifile,mp4file))
    #os.system('ffmpeg -i {} -vcodec libx264 -vf "scale=trunc(iw/3)*2:trunc(ih/3)*2"  {}'.format(avifile, mp4file))
    os.system('ffmpeg -i {} -vcodec libx264 -vf "scale=trunc(iw/ih*150)*2:300"  {}'.format(avifile, mp4file))
    #os.system("rm " + avifile)