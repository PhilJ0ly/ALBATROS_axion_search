# Axion-Like Particle and Dark Photon Search in Low Frequency Radio Band

Philippe Joly

## Abstract
We present a search for dark photons and axion-like particles — well-motivated extensions of the Standard Model that may constitute the elusive dark matter permeating our universe. This paper entails searching for these dark-matter candidates through theorized interactions with the Earth’s ionosphere further constricting the possible mass and coupling of these particles. This effort uses 36 hours of 4 to 11 MHz ALBATROS radio data from McGill Arctic Research Station, re-channelizes it to ∼ 1 Hz resolution, develops a robust integration procedure to suppress unwanted interference and increase the signal of interest’s signature, and finds a reliable statistical test to confirm the presence or absence of a dark matter signal. RESULT TBD yes/no signature characteristic of ALP or DP ionospheric interactions have been detected with some confidence. Significance of observation or breadth of rule-out of mass and coupling.

## Repository Structure
- **albatros_analysis**: Scripts, Classes and Methods designed to process raw ALBATROS radio data on GPU
- **albatros_analysis_cpu**: Scripts, Classes and Methods designed to process raw ALBATROS radio data on CPU
- **Jupyter**: Notebooks to visualise and test scripts concerning ALBATROS data
- **Reports**: Written Reports and Posters associated with the research effort
- **C/CHAIM**: Extraction and processing of A/E-CHAIM data
- **env**: Various BASH scripts to standardize the environmemnt setup on the Trillium GPU and CPU machine

## Key Scripts
 - albatros_analysis/scripts/xcorr/xcorr_gpu.py
 - albatros_analysis/scripts/xcorr/xcorr_gpu_job.sh
 - albatros_analysis/scripts/xcorr/visualise_plasma_bins.py
 - albatros_analysis/scripts/xcorr/helper_gpu.py
 - albatros_analysis/src/utils/pfb_gpu_utils.py
    