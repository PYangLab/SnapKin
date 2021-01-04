#' @export
installSnapkin = function(useGPU=FALSE) {
    # Function for installing conda environment
    if (useGPU) {
        if (Sys.info()["sysname"] == 'Darwin') {
            stop('This function does not work for MacOS. Please use the non-GPU option.')
        }
        reticulate::conda_create(envname='SnapKin-GPU',
                                 packages=c('python=3.8',
                                            'numpy',
                                            'pandas',
                                            'cudatoolkit=10.1',
                                            'cudnn=7.6',
                                            'nccl=2.4',
                                            'tensorflow-gpu=2.2.0'))
        print('Conda environment created: SnapKin-GPU')
    }
    else {
        reticulate::conda_create(envname='SnapKin',
                                 packages=c('python=3.8',
                                            'numpy',
                                            'pandas',
                                            'pip'))
        reticulate::conda_install(envname='SnapKin',
                                  packages='tensorflow==2.2.0',
                                  pip=TRUE)
        print('Conda environment created: SnapKin')
    }
}


