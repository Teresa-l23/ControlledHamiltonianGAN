cd src
python -m hgan.eval \
  --config-path hgan/output/2B_0_1 \
  --output-folder eval_results \
  --override-output-folder
  --system_name two_body 
cd ..