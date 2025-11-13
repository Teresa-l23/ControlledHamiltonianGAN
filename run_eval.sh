cd src
python -m hgan.eval \
  --config-path hgan/configuration.ini \
  --output-folder ../eval_results \
  --system_name pendulum \
  --perplexity 2 5 30 50
cd ..