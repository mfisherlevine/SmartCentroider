import scipy.ndimage as ndimage
from scipy.ndimage.measurements import center_of_mass, maximum_position
from scipy.signal import argrelmax, savgol_filter
import sys
import numpy as np
import new_functions as fn
import pylab as pl

class SmartCentroider(object):
    '''A class for smart centroiding TimepixCam data.
    Gets smart x,y,t centroids from a fileset. A ToF spectrum is generated
    using a subset of the files (evenly sampled from the set)
    to set the band boundaries, though they can be manually specified as well.
    If use_CoM_as_centroid==False, the earliest pixel in the cluster is used,
    otherwise the centre of mass of the cluster is taken as the centroid.'''

    @classmethod
    def DataFrom(cls, data_source):
        self = SmartCentroider(data_source.filelist)
        for k, v in data_source.__dict__.items():
            self.__dict__[k] = v
        return self
    
    def __init__(self, filelist):
        assert len(filelist)>0
        # all variables must be declared here in order to be class variables
        # default values go here, all can be overridden
        
        self.__dict__['filelist'] = filelist
        self.__dict__['n_tof_files'] = 50
        self.__dict__['bands'] = None
        self.__dict__['use_CoM_as_centroid'] = True
        self.__dict__['inc_diagnoal_joins'] = True
        self.__dict__['peak_range'] = 5
        self.__dict__['skiplines'] = 0
        self.__dict__['files_have_bunchIDs'] = False
        self.__dict__['ToF_noise_threshold'] = 2
        self.__dict__['savgol_window_length'] = 5
        self.__dict__['sample_TOF_raw'] = None
        self.__dict__['npix_per_cluster_cut'] = (4,1e9)
        self.__dict__['gaussian_size'] = 1.5
        self.__dict__['sample_TOF_smoothed'] = None
        self.__dict__['peaks'] = []
        self.__dict__['peak_indices'] = []
        self.__dict__['ret'] = {}
        self.__dict__['TMIN'] = 0
        self.__dict__['TMAX'] = 11810
        self.__dict__['main_TOF'] = None
        self.__dict__['DEBUG'] = 1
        self.__dict__['VMI_images'] = []
        self.__dict__['auto_offset'] = 200

        if type(filelist)==str:
            import time
            import cPickle as pickle
            now = time.time()
            pickle_file = open(filelist, 'rb')
            print 'Loading from %s...'%filelist
            class_contents = pickle.load(pickle_file)
            pickle_file.close()
            print 'Loaded in %.2fs from %s'%(time.time()-now, filelist)
            for k, v in class_contents.__dict__.items():
                self.__dict__[k] = v

    def __setattr__(self, attribute, value):
        if not attribute in self.__dict__:
            print "Cannot set %s" % attribute
        else:
            self.__dict__[attribute] = value

    def SaveToPickle(self, filename):
        import time
        import cPickle as pickle
        now = time.time()
        pickle_file = open(filename, 'wb')
        pickle.dump(self, pickle_file, pickle.HIGHEST_PROTOCOL)
        pickle_file.close()
        print 'Saved to %s in %.2fs'%(filename, time.time()-now)


    #########################################################
    #   Real code starts here   #############################
    #########################################################
        
    def MakeSampleTOF(self):
        '''Build a set of files to use for making the ToF.
        The ToF can chage during delay scans etc, so it is
        good to pull from across the range of files rather
        than simply taking the first n files'''
        n_files = len(self.filelist)
        tof_fileset = []
        if self.n_tof_files >= n_files: #if there fewer files that the nominal, take them all
            tof_fileset = self.filelist
        else: 
            indices = [int(np.ceil(i * n_files / self.n_tof_files)) for i in xrange(self.n_tof_files)] #evenly sample the files
            if self.DEBUG >2:print 'Selecting file nums for ToF: %s'%indices
            tof_fileset = [self.filelist[i] for i in indices]
            if self.DEBUG >1: print 'selected %s files for ToF'%len(tof_fileset);sys.stdout.flush()

        tof_imgs = []
        for filename in tof_fileset: # load images
            this_img = fn.TimepixFileToImage(filename, skiplines=self.skiplines, t_min=self.TMIN, t_max=self.TMAX)
            if (this_img==np.zeros((256,256), dtype=np.float)).all()==True: continue # skip completely empty files, as they screw up the minimum range finding code
            tof_imgs.append(this_img)
        maxval = int(np.max(tof_imgs)) # get ranges for histogram
        minval = int(np.min([np.min(_[_>0]) for _ in tof_imgs])) # minumum which is >0 over all images. Will crash on 0-images, but these have been removed before

        ys = np.zeros(((maxval-minval)+1,), dtype=np.int64)
        xs = np.linspace(minval,maxval,(maxval-minval+1)) # make x points for ToF plot
        assert xs[1]-xs[0]==1 #ensure x-axis space is exactly 1

        for img in tof_imgs: # histogram each imgage
            ys += ndimage.histogram(img[img>0],minval,maxval,bins = (maxval-minval)+1) #much faster than pl.hist

        self.sample_TOF_raw = np.zeros((maxval-minval+1,2), dtype=np.int64)
        self.sample_TOF_raw[:,0] = xs
        self.sample_TOF_raw[:,1] = ys

        ys[ys<self.ToF_noise_threshold] = 0 # filter bins with small values

        # redo the ranges after filtering out the small values:
        # TODO: replace this with the bisect function
        first_non_zero, last_non_zero = 0, maxval-minval
        while ys[first_non_zero]<=0:
            first_non_zero += 1
        while ys[last_non_zero]<=0:
            last_non_zero -= 1
            
        xs = xs[first_non_zero:last_non_zero+1] # trim array
        ys = ys[first_non_zero:last_non_zero+1]

        new_ys = savgol_filter(ys,self.savgol_window_length,3) # smooth the ToF spectrum

        self.sample_TOF_smoothed = np.zeros((len(xs),2), dtype=np.int64)
        self.sample_TOF_smoothed[:,0] = xs
        self.sample_TOF_smoothed[:,1] = new_ys

    def CalculateBands(self):
        self.peak_indices = argrelmax(self.sample_TOF_smoothed[:,1], axis=0, order=self.peak_range) # find local maxima, range of 5 each side
        if self.DEBUG>=3: print 'Found %s peaks at %s'%(len(self.peaks), self.peaks); sys.stdout.flush()
        self.peaks = [self.sample_TOF_smoothed[_,0] for _ in self.peak_indices[0]] # Get peak location from indices
        self.bands = [] # generate banks from peaks - need to take the midpoints though!
        minval = np.min(self.sample_TOF_smoothed[:,0])
        maxval = np.max(self.sample_TOF_smoothed[:,0])
        self.bands.append((minval,(self.peaks[0]+self.peaks[1])//2)) #first point to midpoint of first peaks
        for i in xrange(1,len(self.peaks)-1):
            self.bands.append((self.bands[-1][1],(self.peaks[i]+self.peaks[i+1])//2)) # loop through
        self.bands.append((self.bands[-1][1],maxval))# add last midpoint to last value

    def ShowTOF(self, data):
        f = pl.figure(figsize=[12,4]) # Make the figure an appropriate shape
        ax = pl.subplot(111)
        pl.plot(data[:,0],data[:,1],'b') #plot the original ToF
        pl.show()

    def ShowBands(self):
        f = pl.figure(figsize=[12,4]) # Make the figure an appropriate shape
        ax = pl.subplot(111)

        pl.plot(self.sample_TOF_raw[:,0],self.sample_TOF_raw[:,1],'b') #plot the original ToF
        pl.plot(self.sample_TOF_smoothed[:,0],self.sample_TOF_smoothed[:,1],'r')#plot the new, smoothed ToF
        if self.peak_indices!=[]: pl.plot(self.peaks,[-10 for _ in self.peak_indices[0]],'bo') # plot peaks in blue
        pl.plot([_[0] for _ in self.bands],[10 for _ in self.bands],'ro') # plot left band boundaries in red
        pl.plot([_[1] for _ in self.bands],[10 for _ in self.bands],'ro') # plot right band boundaries in red too
        
        ticks = [] # use ticks to display band edges
        for (a, b) in self.bands: # TODO: This could be more elegant
            if a not in ticks: ticks.append(a)
            if b not in ticks: ticks.append(b)

        ticks = sorted(ticks) # don't know if this is necessary but doesn't hurt
        ax.set_xticks(ticks, minor=False)
        ax.xaxis.grid(True, which='major')
        
        # set plots limits
        plot_range = (np.min(self.sample_TOF_smoothed[:,0]),np.max(self.sample_TOF_smoothed[:,0]))
        pl.xlim(plot_range[0]-10,plot_range[1]+10) 

        # TODO: add a legend saying that colours are what
        if self.DEBUG>=1:
            print 'Peaks at %s'%self.peaks
            print 'Bands set at %s'%self.bands;sys.stdout.flush()
        pl.show()
        
        
        
    def FindClusters(self):
        if self.DEBUG>=2: print 'Using %s bands...'%len(self.bands);sys.stdout.flush()

        if self.inc_diagnoal_joins:
            struct_el=[[1,1,1],[1,1,1],[1,1,1]] # for including diagonal connections as well
        else:
            struct_el=[[0,1,0],[1,1,1],[0,1,0]] # for vertical/horizontal connections only

        #     xs, ys, ts = [], [], [] #return lists
        for filenum, filename in enumerate(self.filelist):
            if filenum%500==0: print 'Smart centroided %s of %s files...'%(filenum, len(self.filelist));sys.stdout.flush()
            if self.files_have_bunchIDs: # if we're using bunchIDs then use these as keys in return dict
                fileID = fn.GetBunchIDFromFile(filename)
            else: #otherwise use the filenames
                fileID = filename
            self.ret[fileID]={'xs':[],'ys':[],'ts':[],'npixs':[]}

            img = fn.TimepixFileToImage(filename, skiplines=self.skiplines, t_min=self.TMIN, t_max=self.TMAX)

            segmentation, segments = ndimage.label(img, struct_el) # find clusters
            if self.DEBUG>2: print 'Found %s clusters without using band information'%segments;sys.stdout.flush()
            if self.DEBUG>3: self.DebugPlot(segmentation, 'Segmented image not using bands:')

            seg_sum = 0
            for bandnum,(tmin, tmax) in enumerate(self.bands): # Process each band in turn
                band_img = img.copy() # make a copy
                # TODO: Check that the correct edge is being included/excluded so you're not taking one band edge
                # twice and the other not at all!
                band_img[band_img > tmax] = 0 # threshold the new image
                band_img[band_img <= tmin] = 0
                if self.DEBUG>=3: title = self.DebugPlot(band_img, 'Band %s (%s - %s) image:'%(bandnum, tmin, tmax))

                # find clusters
                segmentation, segments = ndimage.label(band_img, struct_el) 
                seg_sum += segments
                if self.DEBUG>=3: print 'Found %s segs in band %s (%s - %s)'%(segments, bandnum, tmin, tmax);sys.stdout.flush()

                # Get centroids from clusters using specified method:
                if self.use_CoM_as_centroid: # use center of mass weighting
                    com_img = np.zeros_like(band_img) # make an array with ones at all hits, zeros everywhere else
                    com_img[np.nonzero(band_img)] = 1 # otherwise you weight the CoM by the pixel value, not one
                    CoMs = center_of_mass(com_img, segmentation, [_ for _ in xrange(1, segments+1)])

                    # NB do not replace with enumerate, you need to know the actual number for each cluster - think about it.
                    for clust_num in xrange(1, segments+1): # cluster 0 = background so skip it
                        clust_pix_index = np.where(segmentation==clust_num) # find pixels associated with cluster
                        self.ret[fileID]['ts'].append(np.max(img[clust_pix_index]))
                        self.ret[fileID]['xs'].append(CoMs[clust_num-1][0]+0.5) # note these are zero-based from before, so need to -1
                        self.ret[fileID]['ys'].append(CoMs[clust_num-1][1]+0.5)
                        self.ret[fileID]['npixs'].append(len(clust_pix_index[0]))

                else: # Take the earliest timecode in the cluster as the centroid
                    max_positions = maximum_position(band_img, segmentation, [_ for _ in xrange(1,segments+1)])
                    for clust_num in xrange(1, segments+1): # cluster 0 = background so skip it
                        self.ret[fileID]['ts'].append(img[max_positions[clust_num-1]])
                        self.ret[fileID]['xs'].append(max_positions[clust_num-1][0] +.5) 
                        self.ret[fileID]['ys'].append(max_positions[clust_num-1][1] +.5)
                        npix = len(segmentation[segmentation==(clust_num)])
                        self.ret[fileID]['npixs'].append(npix)

                if self.DEBUG>=2: print 'CoMs for band %s: %s'%(bandnum, CoMs);sys.stdout.flush()
            #     index = (np.asarray([_[0] for _ in CoMs]),np.asarray([[_[1] for _ in CoMs]]))
            #     codes = img[CoMs]

            if self.DEBUG>=2: print 'Found %s clusters when using bands'%seg_sum;sys.stdout.flush() 
        print 'Finished smart centroiding %s files...'%(len(self.filelist));sys.stdout.flush()

    
    def MakeVMIsFromBands(self, custom_bands=None, use_gaussians=False, round_centroid_coords=False, only_use_n_files=1e15):
        '''Produce VMI images from centroided clusters for each band defined.
        Custom bands can be provided here, and will not overwrite the main band definitions.
        If using Gaussians, exact centroids are always used.
        If not using Gaussians, if round_centroid_coords==True the nearest pixel is used, otherwise, it is'''
        # TODO: put in a time estimator for this stage as it can be very slow. Might need to loop over band inside files instead of the other way around.
#         import time
        
        if custom_bands is None: custom_bands = self.bands # default to the defined bands if they're not provided
        self.VMI_images = [np.zeros((256,256), dtype=np.float64) for _ in custom_bands] # create images

#         now = time.time()
        for band_num, t_range in enumerate(custom_bands): # loop over bands
            if self.DEBUG>=3:print 'Processing band %s of %s'%(band_num+1, len(self.bands)); sys.stdout.flush()
            #loop over clusters, file by file, using only first n files if specified.
            for filenum, datafilename in enumerate(sorted(self.ret.keys())[:min(only_use_n_files,len(self.ret.keys()))]): 
#                 if filenum == 50:
#                     dt = time.time() - now
#                     print 'Estimated time for completion is another %s seconds...'%
                for x,y,t,npix in zip(self.ret[datafilename]['xs'],
                                      self.ret[datafilename]['ys'],
                                      self.ret[datafilename]['ts'],
                                      self.ret[datafilename]['npixs']):
                    if (npix<self.npix_per_cluster_cut[0]) or (npix>self.npix_per_cluster_cut[1]): continue # apply cluster threshold
                    if t >= t_range[0] and t<t_range[1]: # check cluster is within band
                        if use_gaussians: # add a gaussian if using them
                            self.VMI_images[band_num] += fn.makeGaussian(256,1,self.gaussian_size,[x,y])
                        else: # otherwise, make coordinate integers in the correct way and increment that pixel
                            if round_centroid_coords:
                                x = int(np.round(x,0))
                                y = int(np.round(y,0))
                            else:
                                x = int(x)
                                y = int(y)
                            self.VMI_images[band_num][x][y] += 1

    def BuildMainTOF(self):
        all_ts = []
        for k, v in self.ret.items():
            all_ts.extend(v['ts'])
        print 'Found %s timecodes after centroiding'%len(all_ts)
        all_ts = np.asarray(all_ts)
        minval = int(np.min(all_ts))
        maxval = int(np.max(all_ts))

        ys = np.zeros(((maxval-minval)+1,), dtype=np.int64)
        xs = np.linspace(minval,maxval,(maxval-minval+1)) # make x points for ToF plot
        assert xs[1]-xs[0]==1 #ensure x-axis space is exactly 1

        ys += ndimage.histogram(all_ts,minval,maxval,bins = (maxval-minval)+1)
        
        self.main_TOF = np.zeros((len(xs),2), dtype=np.int64)
        self.main_TOF[:,0] = xs
        self.main_TOF[:,1] = ys

        
    def PrintBandsForEditing(self, one_band_per_line=True):
        '''Print out the band definitions in a format useful for editing.'''
        if one_band_per_line:
            print 'bands = [' + str(self.bands[0]) + ','
            for band in self.bands[1:-1]:
                print '         ' + str(band) + ','
            print '         ' + str(self.bands[-1]) + ']'
        else:
            print 'bands = %s'%self.bands
            
    def DebugPlot(self, data, title=''):
        print title; sys.stdout.flush()
        f = pl.figure(figsize=[8,8]) 
        pl.imshow(data)
        pl.show()
        
                      
    def ShowAllVMIs(self, vmin=None, vmax=None, cmap='jet', white_background=False, white_background_threshold=0.1, save_path=None):
        for im_num, image in enumerate(self.VMI_images):
            if save_path:
                savefig = save_path + 'VMI_%s.png'%im_num
            else:
                savefig = None
            self.ShowVMIimage(image, vmin=vmin, vmax=vmax, cmap=cmap,  white_background=white_background, white_background_threshold=white_background_threshold, savefig=savefig)
                            
    def ShowVMIimage(self, image, vmin=None, vmax=None, cmap='jet', title = '', savefig='', white_background=False, white_background_threshold=0.1):
        import numpy as np
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig = pl.figure(figsize = [10,10])
        ax = fig.add_subplot(111)

        if vmax == 'auto':
            element = (256*256) - self.auto_offset - 1
            tmp = image.flatten()
            tmp.sort()
            vmax = tmp[element]
            vmin = tmp[self.auto_offset]
            print 'Auto vmax = %s, real max = %s'%(vmax, np.max(image))

        if vmin == 'auto':
            tmp = image.flatten()
            vmin = min(_ for _ in tmp if _ > 0)
            print 'Auto vmin = %s'%vmin

        display_image = image.copy()
        if white_background:
            vmin = max(vmin, white_background_threshold)
            display_image[display_image<=white_background_threshold]=np.nan

        im = ax.imshow(display_image, vmin=vmin, vmax=vmax, cmap=cmap, interpolation='nearest')
        ax.set_title(title)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        pl.colorbar(im, cax=cax)        
        
        if savefig: fig.savefig(savefig)
        return im  

#     def ShowVMIimage(self, image, title=''):
#         print title; sys.stdout.flush()
#         f = pl.figure(figsize=[8,8])
#         pl.imshow(image)
#         pl.show()
    def GetClusterSizeList(self):
        cluster_sizes = []
        for k, v in self.ret.items():
            cluster_sizes.extend(v['npixs'])
        return cluster_sizes

    def PlotClusterSizeHistogram(self):
        import pylab as plt
        cluster_sizes = GetClusterSizeList(self)
        plt.figure(figsize=(12,4))
        plt.hist(cluster_sizes, bins = 50, range=(0,50)) # hardcoded, with binsize==1

    def run(self):
        self.MakeSampleTOF()
        if self.DEBUG>=2:
            print 'Raw ToF spectrum for sample files'
            self.ShowTOF(self.sample_TOF_raw)
        if self.DEBUG>=1:
            print 'Smoothed, noise suppressed and truncated ToF spectrum for sample files'
            self.ShowTOF(self.sample_TOF_smoothed)
        
        if not self.bands:
            self.CalculateBands()
        if self.DEBUG>=0: self.ShowBands()
    
        self.FindClusters()
        self.BuildMainTOF()
        self.MakeVMIsFromBands()
        self.ShowAllVMIs()

        # return self.ret.copy()
