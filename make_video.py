import argparse
import tensorflow as tf
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from PIL import Image
import re
import collections

def get_prob_maps(probs):
    prob_map = probs[:,:,:,0]-probs[:,:,:,1]
    prob_map+=1
    prob_map/=2
    prob_map*=100
    return np.round(prob_map)

def plot_entropy(x00,yy,prob_maps,entropy,out_dir='test_results',i0=0):
    count=0
    for i in range(5):

            plt.figure(dpi=600)
            plt.subplot(1,2,1)
            plt.imshow(x00[i])
            plt.axis('off')

            plt.subplot(1,2,2)

            plt.imshow(entropy[i],**{'cmap':'gray'})
            plt.axis('off')

            plt.show()
            plt.tight_layout()

            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                        hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())


            plt.savefig(out_dir+'/'+str(i+i0)+'.jpg',bbox_inches='tight',pad_inches=0)
            plt.close()

            count+=1

def plot_probs(x00,yy,prob_maps,entropy,out_dir='test_results',i0=0):
    count=0
    for i in range(5):

            plt.figure(dpi=600)
            plt.subplot(1,2,1)
            plt.imshow(x00[i])
            plt.axis('off')

            plt.subplot(1,2,2)
            plt.imshow(prob_maps[i],**{'cmap':'inferno'})
            plt.axis('off')

            plt.show()
            plt.tight_layout()

            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                        hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())


            plt.savefig(out_dir+'/'+str(i+i0)+'.jpg',bbox_inches='tight',pad_inches=0)
            plt.close()

            count+=1

def plot_masks(x00,yy,prob_maps,entropy,out_dir='test_results',i0=0):
    count=0
    for i in range(5):

            kp = np.argwhere(-1*(yy[i]-1))

            plt.figure(dpi=600)

            plt.subplot(1,2,1)
            plt.imshow(x00[i])
            plt.axis('off')

            plt.subplot(1,2,2)
            plt.imshow(x00[i])
            plt.scatter(kp[:,1],kp[:,0],c='r',s=0.2,alpha=.5)
            plt.axis('off')

            plt.tight_layout()

            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                        hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())


            plt.savefig(out_dir+'/'+str(i+i0)+'.jpg',bbox_inches='tight',pad_inches=0)
            plt.close()

            count+=1

def plot(x00,yy,prob_maps,entropy,out_dir='test_results',i0=0):
    count=0
    for i in range(5):

            kp = np.argwhere(-1*(yy[i]-1))

            plt.figure(dpi=600)

            plt.subplot(1,4,1)
            plt.imshow(x00[i])
            plt.axis('off')

            plt.subplot(1,4,2)
            plt.imshow(x00[i])
            plt.scatter(kp[:,1],kp[:,0],c='r',s=0.05,alpha=.15)
            plt.axis('off')


            plt.subplot(1,4,3)
            plt.imshow(prob_maps[i],**{'cmap':'inferno'})
            plt.axis('off')

            plt.subplot(1,4,4)
            plt.imshow(entropy[i],**{'cmap':'gray'})
            plt.axis('off')

            plt.show()
            plt.tight_layout()

            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                        hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())


            plt.savefig(out_dir+'/'+str(i+i0)+'.jpg',bbox_inches='tight',pad_inches=0)
            plt.close()

            count+=1


def load_images(input_folder,size):
    imgs = os.listdir(input_folder)
    sorted_img = {}
    for i in imgs:
        for x in re.split('(\d+)',i):
            if x.isdigit():
                sorted_img[int(x)]=i

    img_stack = []
    od = collections.OrderedDict(sorted(sorted_img.items()))
    for k, v in od.items():
        img = Image.open(os.path.join(input_folder,v))
        img = img.resize((size,size),Image.ANTIALIAS)
        img = np.expand_dims(np.array(img),axis=0)
        img_stack.append(img)
    return img_stack

def main(input_folder,model_meta_file,chkpt_dir,output_folder,size,plt_all=1,plt_masks=1,plt_probs=1,plt_entropy=1):

    img_stack = load_images(input_folder,size)

    sess = tf.Session()
    new_saver = tf.train.import_meta_graph(model_meta_file)
    new_saver.restore(sess, tf.train.latest_checkpoint(chkpt_dir))

    graph = tf.get_default_graph()

    op_to_restore = graph.get_tensor_by_name('logits:0')

    x0 = graph.get_tensor_by_name('x0:0')

    n_batches = len(img_stack)//5

    b1 = 5
    b0= 0
    count = 1
    for i in range(n_batches):
        print('batch %d of %d'%(i,n_batches))
        x00=np.vstack(img_stack[b0:b1])/255.0
        N=5

        y = sess.run(op_to_restore,feed_dict={x0:x00})

        yy = np.argmax(y,axis=1).reshape((N,size,size))
        probs = (np.exp(y)/np.sum(np.exp(y),axis=1)[:,np.newaxis])
        probs = probs.reshape((N,size,size,2))

        prob_maps = get_prob_maps(probs)

        entropy = np.sum(-np.log(probs)*probs,axis=3)
        entropy = entropy.reshape((N,size,size))


        if plt_all==1:plot(x00,yy,prob_maps,entropy,out_dir=output_folder+'/plots',i0=b0)
        if plt_masks==1:plot_masks(x00,yy,prob_maps,entropy,out_dir=output_folder+'/masks',i0=b0)
        if plt_probs==1:plot_probs(x00,yy,prob_maps,entropy,out_dir=output_folder+'/probs',i0=b0)
        if plt_entropy==1:plot_entropy(x00,yy,prob_maps,entropy,out_dir=output_folder+'/entropy',i0=b0)

        b0+=5
        b1+=5


if __name__ == "__main__":
    size = 256

    parser = argparse.ArgumentParser(description='Hyperparameters!')
    parser.add_argument('-i','--input_dir',help='Input folder, path2imgs',default='video/imgs')
    parser.add_argument('-m','--model',help='model_meta_files',default='tmp/model.ckpt.meta')
    parser.add_argument('-c','--check_points',help='Model saved checkpoints',default='tmp')
    parser.add_argument('-o','--output_dir',help='Output folder',default='video/processed_imgs3')
    parser.add_argument('-en','--plt_entropy',help='Build entropy video (1 or 0)',type=int,default=1)
    parser.add_argument('-ms','--plt_masks',help='Build masks video (1 or 0)',type=int,default=1)
    parser.add_argument('-pr','--plt_probs',help='Build probs video (1 or 0)',type=int,default=1)
    parser.add_argument('-all','--plt_all',help='Build all_plots video (1 or 0)',type=int,default=1)

    args = vars(parser.parse_args())

    os.system('mkdir '+args['output_dir'])
    if args['plt_all']==1:os.system('mkdir '+args['output_dir']+'/plots')
    if args['plt_entropy']==1: os.system('mkdir '+args['output_dir']+'/entropy')
    if args['plt_masks']==1: os.system('mkdir '+args['output_dir']+'/masks')
    if args['plt_probs']==1: os.system('mkdir '+args['output_dir']+'/probs')

    main(input_folder=args['input_dir'],model_meta_file=args['model'],chkpt_dir=args['check_points'],output_folder=args['output_dir'],size=size)
    os.chdir(args['output_dir'])
    os.system('ffmpeg -f image2 -i plots/%d.jpg allvideo.avi -y')
    os.system('ffmpeg -f image2 -i '+'entropy/%d.jpg entropyvideo.avi -y')
    os.system('ffmpeg -f image2 -i '+'masks/%d.jpg masksvideo.avi -y')
    os.system('ffmpeg -f image2 -i '+'probs/%d.jpg probsvideo.avi -y')
