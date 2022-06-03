# PMixUp

--- 
Implementation of paper "PMixUp: Simultaneous Utilization of Part-of-Speech Replacement and Feature Space Interpolation for Text Data Augmentation


## Paper Abstract
---
Data augmentation methods have become a de facto technique in various NLP tasks to overcome the lack of a large-scale, qualified training set. The previous studies presented several data augmentation methods, such as replacing tokens with synonyms or interpolating feature space of given text input. While they are known to be convenient and promising, several limits exist. First, prior studies simply treated topic classification and sentiment analysis under the same category of text classification while we presume they have distinct characteristics. Second, previously-proposed replacement-based methods bear several improvement avenues as they utilize heuristics or statistical approaches for choosing synonyms. Lastly, while the feature space interpolation method achieved current state-of-the-art, prior studies have not comprehensively utilized it with replacement-based methods. To mitigate these drawbacks, we first analyzed which POS tags are important in each text classification task, and resulted that nouns are essential to topic classification, while sentiment analysis regards verbs and adjectives as important POS information. Contrary to the aforementioned analysis, we discover that augmenting verbs and adjective tokens commonly improves text classification performance regardless of its type. Lastly, we propose PMixUp, a novel data augmentation strategy that simultaneously utilizes replacement-based and feature space interpolation methods. We empirically examine that they are new state-of-the-art in nine public benchmark settings, especially under the few training samples. 


## How to run 
```bash
$cd src
```

---
### Ratio of Top-12 POS tags existing in the 9 benchmark datasets (Table 1)

```bash
$ python inspect_pos.py
```

### Ratio of Nouns, Verbs, Adjs in probability-based important tokens for each dataset (Table 2)

```bash
$ python pos_in_important.py
```

### Classification performance after removing tokens with each POS (Table 3)

```bash
$ python removed_classification.py
```

### Classification performance after augmenting tokens with each POS (Table 4)
```bash
$ python pos_augmentation.py
```

### PMixUp 
```bash
$ python pmixup.py
```