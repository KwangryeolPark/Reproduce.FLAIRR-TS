mkdir -p dataset/ETT-small
mkdir -p dataset/electricity
mkdir -p dataset/traffic
mkdir -p dataset/weather

(
  cd dataset

  # -t 5: Retry up to 5 times on failure
  # -nc: Skip download if file already exists (no-clobber)
  # -q: Run quietly to suppress long logs (remove if not needed)
  COMMON_ARGS="-t 5 -nc"
  URL_PREFIX="https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main"
  
  wget $COMMON_ARGS $URL_PREFIX/ETTh1.csv &
  wget $COMMON_ARGS $URL_PREFIX/ETTh2.csv &
  wget $COMMON_ARGS $URL_PREFIX/ETTm1.csv &
  wget $COMMON_ARGS $URL_PREFIX/ETTm2.csv &
  wget $COMMON_ARGS $URL_PREFIX/PEMS03.npz &
  wget $COMMON_ARGS $URL_PREFIX/PEMS04.npz &
  wget $COMMON_ARGS $URL_PREFIX/PEMS07.npz &
  wget $COMMON_ARGS $URL_PREFIX/PEMS08.npz &
  wget $COMMON_ARGS $URL_PREFIX/electricity.csv &
  wget $COMMON_ARGS $URL_PREFIX/exchange_rate.csv &
  wget $COMMON_ARGS $URL_PREFIX/solar_AL.pkl &
  wget $COMMON_ARGS $URL_PREFIX/solar_AL.txt &
  wget $COMMON_ARGS $URL_PREFIX/traffic.csv &
  wget $COMMON_ARGS $URL_PREFIX/weather.csv &
  wget $COMMON_ARGS $URL_PREFIX/national_illness.csv &
  
  # Since illness.csv will be renamed, only download if the final file (national_illness.csv) does not exist
  if [ ! -f "national_illness.csv" ]; then
      wget $COMMON_ARGS $URL_PREFIX/illness.csv &
  fi
  
  wait

  # Rename illness.csv only if it was downloaded
  if [ -f "illness.csv" ]; then
      mv illness.csv national_illness.csv
  fi

  # For ETT series
  cd ETT-small
  # Use -f (force) to prevent errors if symbolic links already exist
  ln -sf ../ETT*.csv .
)

(
    cd dataset/electricity
    ln -sf ../electricity.csv .
)
(
    cd dataset/traffic
    ln -sf ../traffic.csv .
)
(
    cd dataset/weather
    ln -sf ../weather.csv .
)

wait
