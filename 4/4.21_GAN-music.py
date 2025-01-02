"""
Пример на Python, который использует библиотеку Magenta для создания
музыкальной генеративной модели с помощью глубокого обучения.
"""

# https://storage.googleapis.com/magentadata/models/music_vae/checkpoints/cat-mel_2bar_big.tar

import magenta.music as mm
import magenta.models.music_vae as mvae
from magenta.models.music_vae import TrainedModel
from magenta.models.shared import sequence_generator_bundle
from magenta.models.shared import sequence_generator as sg

BASE_DIR = "gs://download.magenta.tensorflow.org/models/music_vae/colab2"

# Загрузка предобученной модели
model_name = "cat-mel_2bar_big"
model_config = mvae.configs.CONFIG_MAP[model_name]
model = TrainedModel(model_config, batch_size=4,
                     checkpoint_dir_or_path=BASE_DIR + '/checkpoints/mel_2bar_big.ckpt')

# Создание среды обучения
mm.notebook_utils.download_bundle(f"{model_name}.mag", "bundles")
bundle = sequence_generator_bundle.read_bundle_file(f"bundles/{model_name}.mag")
generator_map = mm.generator_pb2.GeneratorDetails(id='default', description='')
generator = sg.BaseSequenceGenerator(model, generator_map)

# Генерация новой музыки
num_steps = 128  # Длина новой мелодии в шагах
temperature = 0.5  # Параметр для контроля степени случайности генерации
generated_sequence = generator.generate(num_steps=num_steps, temperature=temperature)

# Сохранение сгенерированной музыки в MIDI файл
mm.sequence_proto_to_midi_file(generated_sequence, 'generated_music.mid')
