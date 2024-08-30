# Go to docs for Usage and model list


# Updates
```
2023.08.09 - Add TensorRTStableDiffusionFillInpaintPipeline @olnayoung
2023.07.19 - Add ControlNetFillInpaintPipeline
2023.07.18 - Add TensorRTStableDiffusionInpaintPipeline, StableDiffusionFillInpaintPipeline
2023.01.13 - Updated related diffusers version to 0.25.0
```

# Install by pip
```
pip install git+https://github.com/team-ailab/est_models.git
```

# Install by poetry
```bash
#if necessary
poetry config virtualenvs.create false

git clone https://github.com/team-ailab/est_models.git
cd est_models
poetry install
```

# Do pre-commit
```bash
pip install pre-commit
pre-commit install
```

# Documents
- Check docs folder

# Contributions
```
#RUN

python -m pytest tests
```
