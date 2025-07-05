# TranslitASR-KWS 
## Multilingual Query-by-Example KWS for Indian Languages using Transliteration

- **Fairseq training** follows the same method used in [AI4Bharat/IndicWav2Vec](https://github.com/AI4Bharat/IndicWav2Vec); please follow the setup instructions in that repo.

- The **config** and **manifest** files required to run the above recipe for Transliteration ASR-KWS are provided in this repository.  

- **Training files** (Kathbath) and **test files** (IndicSUPERB QbE eval) [AI4Bharat/IndicSUPERB](https://github.com/AI4Bharat/IndicSUPERB).

- The `train.sh` script invokes the Fairseq ASR training command, using manifest files that list the transliterated Kathbath audio data required to train the Transliteration ASR model. 

-  Both manifest files (containing the transliterated Devanagiri text) and the trained Transliteration ASR-KWS model (`mr-pairs`) can be downloaded from [Google Drive](https://drive.google.com/drive/folders/12hlXDW1uHe0Eakw0aaRpQLNrGAdrYpDK?usp=sharing). Edit the manifest files so that the audio filepath point to your local Kathbath audio file locations.

- VAD is applied on the IndicSUPERB QbE eval audio files before evaluation.  
  ```python
  python qbe_vad.py
  ```

- **Inference** can be performed by running:
  ```bash
  bash infer.sh
  ```
The infer.sh script uses the Transliteration ASR-KWS model to extract embeddings from the test set, runs DTW between the reference Audio and eval_queries segments, and then computes the final retrieval scores.

-  The provided model's MTWV scores:
    | Language  | maxTWV |
    | --------- | ------ |
    | Tamil     | 0.511  |
    | Telugu    | 0.374  |
    | Bengali   | 0.391  |
    | Gujarati  | 0.542  |
    | Hindi     | 0.655  |
    | Kannada   | 0.669  |
    | Malayalam | 0.353  |
    | Marathi   | 0.517  |
    | Odia      | 0.481  |
    | Punjabi   | 0.575  |
    | **Average** | **0.507** |


