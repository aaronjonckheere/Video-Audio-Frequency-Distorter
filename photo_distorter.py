import numpy as np
import cv2
import scipy
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian
from scipy.io import wavfile
import matplotlib.pyplot as plt
import math
from numba import jit
import ffmpeg


# 
photo_file = "Image-01-2.jpg"
audio_file = "Skrillex - Scary Monsters And Nice Sprites (Official Audio).wav"
int_video_name = "test.mp4"
output_name = "output.mp4"

#STFT Function Parameters
overlap = 20000
fps = 24.0

# Transform Function Parameters
scaling = 1
after_scaling = 1
power = 2

@jit(nopython=True)
# Filter high frequencies' effect
def frequency_map(f,sensitivity,offset = 0):
    return (.5 - .5*math.tanh((f-offset)/sensitivity)) 

@jit(nopython=True)
def transform_function_standard(pixel, left, right, i, j, length):
    return after_scaling * pixel * (1 - (scaling * left[i]* frequency_map(i,500) * right[i]* frequency_map(j,500)))**power

@jit(nopython=True)
def transform_function_filter(pixel, left, right, i, j, lmax, rmax):
    return after_scaling * pixel * ((scaling * frequency_map(i, 10000, lmax) * frequency_map(j, 10000, rmax)) ** power)


@jit(nopython = True)
def photo_fft(photo_transformed, photo_transformed_output, audio_fft_l, audio_fft_r, transform_function = transform_function_standard):
    width, height = photo_transformed.shape
    width_scale = len(audio_fft_l)/width
    height_scale = len(audio_fft_r)/height
    audio_fft_l /= np.max(np.abs(audio_fft_l))
    audio_fft_r /= np.max(np.abs(audio_fft_r))
    for i in range(photo_transformed_output.shape[0]):
        for j in range(photo_transformed_output.shape[1]):
            photo_transformed_output[i,j] = transform_function(photo_transformed[i,j], audio_fft_l, audio_fft_r,
                                                        int(i*width_scale), int(j*height_scale), len(audio_fft_l))
    return photo_transformed_output

def spec(audio, rate):
    overlap = 20000
    fps = 24.0
    win_size = int(rate/fps + overlap)
    g_std = int(win_size/25)
    w = gaussian(win_size, std=g_std, sym=True)
    SFT = ShortTimeFFT(w, int(rate/fps), rate, scale_to="psd")
    Sx2 = SFT.spectrogram(audio[:,0]) 
    fig1, ax1 = plt.subplots(figsize=(6., 4.)) 
    N= len(audio[:,0]) # enlarge plot a bit

    t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot

    ax1.set_title(rf"Spectrogram ({SFT.m_num*SFT.T:g}$\,s$ Gaussian " +

                rf"window, $\sigma_t={g_std*SFT.T:g}\,$s)")

    ax1.set(xlabel=f"Time $t$ in seconds ({SFT.p_num(N)} slices, " +

                rf"$\Delta t = {SFT.delta_t:g}\,$s)",

            ylabel=f"Freq. $f$ in Hz ({SFT.f_pts} bins, " +

                rf"$\Delta f = {SFT.delta_f:g}\,$Hz)",

            xlim=(t_lo, t_hi))

    Sx_dB = 10 * np.log10(np.fmax(Sx2, 1e-4))  # limit range to -40 dB

    im1 = ax1.imshow(Sx_dB, origin='lower', aspect='auto',

                    extent=SFT.extent(N), cmap='magma')

    ax1.plot(1/rate, 1000, 'g--', alpha=.5, label='$f_i(t)$')

    fig1.colorbar(im1, label='Power Spectral Density ' +

                            r"$20\,\log_{10}|S_x(t, f)|$ in dB")


    # Shade areas where window slices stick out to the side:

    for t0_, t1_ in [(t_lo, SFT.lower_border_end[0] * SFT.T),

                    (SFT.upper_border_begin(N)[0] * SFT.T, t_hi)]:

        ax1.axvspan(t0_, t1_, color='w', linewidth=0, alpha=.3)

    for t_ in [0, N * SFT.T]:  # mark signal borders with vertical line

        ax1.axvline(t_, color='c', linestyle='--', alpha=0.5)

    ax1.legend()

    fig1.tight_layout()

    plt.show()


def main():
    rate, audio = wavfile.read(audio_file)
    #spec(audio,rate)
    win_size = int(rate/fps + overlap)
    g_std = int(win_size/25)
    w = gaussian(win_size, std=g_std, sym=True)
    SFT = ShortTimeFFT(w, int(rate/fps), rate, scale_to="psd")
    left_chan_fft = SFT.stft(audio[:,0])
    right_chan_fft = SFT.stft(audio[:,1])
    photo = cv2.imread(photo_file)
    writer = cv2.VideoWriter(int_video_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (photo.shape[1], photo.shape[0]), True)
    photo_transformed = [np.fft.rfft2(photo[:,:,k]) for k in range(3)]
    photo_transformed_output = [np.fft.rfft2(photo[:,:,k]) for k in range(3)]
    for x in range(2400): 
        audio_l = left_chan_fft[:,x]
        audio_r = right_chan_fft[:,x]
        for k in range(3):
            photo_transformed_output[k] = photo_fft(photo_transformed[k], photo_transformed_output[k], audio_l, audio_r)
            photo[:,:,k] = np.fft.irfft2(photo_transformed_output[k])
        writer.write(photo.astype('uint8'))
        cv2.imshow("test", photo)
        cv2.waitKey(1)
    writer.release()
    input_video = ffmpeg.input(int_video_name)
    input_audio = ffmpeg.input(audio_file)
    out = ffmpeg.output(input_video, input_audio, output_name, vcodec='libx264', acodec='aac', strict='experimental' )
    out.run()



if __name__ == "__main__":
    main()

