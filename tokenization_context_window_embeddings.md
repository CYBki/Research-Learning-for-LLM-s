# Bu doküman kimin için?
Bu rehber, **module1_1.ipynb** not defterini temel alarak *sıfırdan* dil modellemenin en temel yapıtaşlarını öğrenmek isteyenler içindir. İçerikte notebook’taki akışı adım adım açıklar; kod parçacıkları ve görseller hangi kavramı somutladığını belirterek anlatılır.

> Hedef: Bir metni nasıl **token**’lara çeviririz, bu token’lardan **girdi/çıktı örnekleri** (context windows) nasıl oluşturulur, **embedding** nedir ve gerçek bir modelin (Gemma) token gömme katmanını nasıl gözlemleriz — bunları kavramak.

---

## Gereksinimler
- Python 3.10+
- Kütüphaneler: `torch`, `transformers`, `plotly`, (notebook’ta ayrıca yerel modüller: `tokenizer.py`, `text_dataset.py`)
- Yerel dosyalar: `text.txt` (ham metin), `tokenizer.json` (sözlük/kurallar), opsiyonel: `token_ids.txt` (çıktı)

> **Not:** Notebook’ta `Tokenizer` (yerel sınıf) ve `create_data_loader` (yerel fonksiyon) kullanılıyor. Aşağıda bu bileşenlerin *ne yaptığı* açıklanır, böylece eşdeğerlerini kolayca yazabilir/yerine koyabilirsiniz.

---

## 1) Dil modellemenin özü: “Bir sonraki token”
Notebook şu sezgiyi kurarak başlıyor:

- Örnek metin: `"the capital of france is paris"`
- Örnek istem (prompt): `"the capital of the united states is not "`

Amaç: Model, **bağlamı** (önceki token’lar) alıp, bir sonraki makul token’ı tahmin eder. Bu mantık daha sonra girdi/çıktı çiftleri kurarken doğrudan kullanılır.

**Pad ve hedef kaydırma (shift):**
- `data = "the capital <pad> <pad> ..."`
- `hedef = "capital of <pad> ..."`

Burada hedef, girdinin **bir token sağa kaydırılmış** halidir. Yani model, *her konumda bir sonraki token’ı* tahmin etmeyi öğrenir.

---

## 2) Context length (bağlam uzunluğu)
Notebook’ta `context_length = 12` gibi bir sabit var. Bu, modele her adımda verilecek **maksimum token sayısı**dır (pencere genişliği).

- Uzun metinleri işlerken, bu genişlikte **kaydırmalı pencereler** ile ilerlenir (stride).
- Büyük modellerin (ör. GPT-4o, Gemma) kendi üst sınırları vardır; burada kavramsal olarak küçük bir değerle çalışıyoruz.

---

## 3) Tokenization (metni sayılara dönüştürme)
Notebook’ta iki tip tokenizer görülüyor:

1. **Yerel Tokenizer** (`from tokenizer import Tokenizer`):
   - `tokenizer = Tokenizer("tokenizer.json")`
   - `ids = tokenizer.encode(prompt)`
   - Beklenen arayüz: `encode(text) -> List[int]`, `tokenize(text) -> List[str]`, opsiyonel `decode(ids) -> str`.
   - `tokenizer.json`, genelde sözlük (token->id) ve kuralları içerir (ör. BPE/WordPiece).

2. **Hugging Face Tokenizer** (`AutoTokenizer`):
   - `gemma_tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")`
   - Aynı metni encode edip gerçek bir modelin token id’lerini üretir.

> **İpucu:** Eğer kendi `Tokenizer`’ınızı yazıyorsanız, en basit haliyle boşlukta bölüp bir sözlük oluşturabilir, OOV (sözlük dışı) için `<unk>` id’i ayırabilirsiniz. Üretimde ise BPE/WordPiece gibi alt-parça tokenizasyonu şarttır.

---

## 4) Ham metni okuyup token id’lerine dökme
Notebook akışı:

- `text.txt` içeriği okunur.
- `token_ids = tokenizer.encode(text)` ile tüm metin id’lere çevrilir.
- Bu id’ler `token_ids.txt` olarak kaydedilir (isteğe bağlı).

Bunun amacı: **Tek kaynak metinden**, eğitim için kullanılacak sayısal temsilin üretilmesi.

---

## 5) Eğitim verisi pencereleri (context windows) — DataLoader mantığı
Notebook’ta:

```python
from text_dataset import create_data_loader
stride = 12
train_data_loader = create_data_loader(token_ids, context_length, stride, batch_size=1, shuffle=False)
```

Bu fonksiyon tipik olarak şunları yapar:
- `token_ids` üzerinde `context_length` genişliğinde **pencereler** oluşturur.
- Her pencere için **girdi**: ilk `context_length` token.
- **Hedef**: girdiyle aynı dizinin *bir sağa kaydırılmış* hali (son token hedefte kullanılmaz, yerine sonraki gerçek token gelir; eğer yoksa `<pad>`/`<eos>` konur).
- `stride`, pencerenin bir sonrakine kayma miktarıdır (ör. 12 ise, ardışık pencereler çakışmaz; 1 olsaydı her adımda 1 token kayardı).

Örnek kullanım:
```python
for step, batch in enumerate(train_data_loader):
    x, y = batch  # x: [B, context_length], y: [B, context_length]
    # model(x) -> logits; loss = CrossEntropy(logits, y)
    if step > 2:  # sadece örnek
        break
```

> **Sık hata:** Hedef diziyi aynı girdi dizisiyle *aynı hizaya* koymak. Doğrusu, hedefin **1 sağa kaymış** olmasıdır. Aksi halde model, “aynı token’ı” tahmin etmeyi öğrenir.

---

## 6) Embedding nedir? (Sayısal anlamlar)
Notebook iki seviyede embedding gösteriyor:

1) **Elle kurulmuş küçük vektörler**
   - Örn. 4 boyutlu vektörlerle `"the", "capital", "of", "united", "states"` için sayısal karşılıklar tanımlanır.
   - Amaç: Her kelimenin bir **anlam noktası** olduğuna dair sezgi kazandırmak. Bu noktalar çok boyutlu uzayda yer alır.

2) **PyTorch Embedding katmanı**
   ```python
   import torch
   embeddings = torch.nn.Embedding(num_embeddings=64, embedding_dim=4)
   tokens = tokenizer.encode(sentence)
   meanings = embeddings(torch.tensor(tokens))  # [seq_len, 4]
   ```
   - Bu katman, her id için öğrenilebilen bir vektör döndürür. Eğitim sırasında bu vektörler **görev sinyaline göre** güncellenir.

> Gerçek modellerde embedding boyutu yüzlerce/ binlerce olabilir (örn. 2048). Notebook’ta basitlik için 4 boyut ve örnek eksen adları (“Sertlik/Parlaklık/Kırmızılık”) verilmiş.

---

## 7) Gemma (HF) ile gerçek token gömmeleri
Notebook’ta gerçek bir modelin token gömme katmanı çağrılıyor:

```python
from transformers import AutoTokenizer, Gemma3ForCausalLM

gemma_tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
gemma_model = Gemma3ForCausalLM.from_pretrained("google/gemma-3-1b-it")

sentence = "the capital of united states and the capital of france"
ids = gemma_tokenizer.encode(sentence)
emb = gemma_model.model.embed_tokens(torch.tensor(ids))  # [seq_len, hidden_size]
```

- `embed_tokens`, modelin kelime/alt-parça id’lerini **yüksek boyutlu** (modelin gizli boyutu) vektörlere çevirir.
- Daha sonra bu vektörlerin **3 boyutlu bir izdüşümü** seçilerek (ör. `dims=[20,21,22]`) Plotly ile görselleştirilir. Bu, *gerçek* embedding uzayının sadece küçük bir kesitidir.

> **Uyarı:** Büyük modelleri indirirken RAM/VRAM ve ağ kullanımına dikkat edin. Sadece tokenizer kullanmak isterseniz, modeli indirmeden `AutoTokenizer` ile yetinebilirsiniz.

---

## 8) Embedding’leri görselleştirme (Plotly)
Notebook, şu fonksiyon gibi bir 3B serpiştirme grafiği çizer:
- Girdi: `words` (şekil `[seq_len, dim]`), `labels` (token string’leri), `color`.
- Çıkış: Etiketli 3B noktalar.

Bu görselleştirme size şunu sezdirir:
- Aynı cümledeki bazı alt-parçalar/kelimeler, uzayda birbirine **yakın** konumlanabilir (paylaşılan dağıtımsal özellikler).
- Boyut sayısı çok büyük olduğundan, seçtiğiniz 3 eksen **keyfi bir görünüm** verir; tüm yapıyı temsil etmez.

---

## 9) Parçaları birleştirme: Uçtan uca mini akış
1. **Metni hazırla:** `text.txt` içine veri koyun.
2. **Tokenize et:** Yerel `Tokenizer` ile tüm metni id’lere çevirin ve gerekirse `token_ids.txt` olarak kaydedin.
3. **Pencereleri üret:** `create_data_loader(token_ids, context_length, stride, ...)` çağrısı ile `(x, y)` çiftlerini üretin (y, x’in 1 sağa kaymış hali).
4. **Embedding’i kontrol et:** İster `nn.Embedding`, ister Gemma’nın `embed_tokens`’ını kullanarak id’leri vektörlere çevirin.
5. **Görselleştir:** Plotly ile 3B serpiştirme grafiklerinde token konumlarını inceleyin.

> Buraya bir **eğitim döngüsü** ekleyerek (optimizer, `CrossEntropyLoss`, `model.forward`) gerçek bir dil modeli eğitiminin ilk basamağına geçebilirsiniz.

---

## 10) Sık yapılan hatalar
- **Hedef kaydırmasını unutmak:** `y` dizisini `x` ile aynı yapmak.
- **Yanlış `stride`:** Çok büyük stride, veri çeşitliliğini azaltır; çok küçük stride, tekrar oranını artırır (ancak daha zengin örnek üretir).
- **Tokenizer uyumsuzluğu:** Yerel tokenizer ile HF tokenizer’ı karıştırmak; id alanları farklıdır.
- **Görselleştirme yanılgısı:** 3 boyut, yüksek boyutun küçük kesiti; “uzaktalar, demek ki ilgisiz” gibi kesin yargılardan kaçının.

---

## 11) Mini alıştırmalar
- **A1.** Aynı cümlede `"france"` yerine `"germany"` kullanıp embedding noktalarının yer değiştirmesini gözlemleyin.
- **A2.** `context_length` ve `stride` değerlerini değiştirip üretilen `(x, y)` örneklerinin sayısını karşılaştırın.
- **A3.** `nn.Embedding(…, embedding_dim=8)` ile boyutu artırıp, farklı 3 eksen dilimlerini (`dims=[0,1,2]`, `[10,11,12]` gibi) görselleştirin.
- **A4.** Küçük bir `CrossEntropyLoss` deneyi kurun: Rastgele başlatılmış küçük bir tek-katmanlı dil modeliyle 1–2 adım eğitim yapıp loss’un düştüğünü gözlemleyin.

---

## 12) Referans/Şablon kod parçaları
Aşağıda notebook’taki işlevleri dışarıda yeniden kurarken kullanabileceğiniz kısa şablonlar verilmiştir.

**Basit pencereleyici (DataLoader benzeri) — eğitim seti üretimi**
```python
# Minimal example – English comments
from typing import List, Tuple
import torch

def make_windows(token_ids: List[int], ctx_len: int, stride: int) -> List[Tuple[List[int], List[int]]]:
    windows = []
    for start in range(0, max(0, len(token_ids) - ctx_len), stride):
        x = token_ids[start:start+ctx_len]
        # target is the next-token version of x
        y = token_ids[start+1:start+ctx_len+1]
        if len(y) < ctx_len:
            # pad target if you use padding; or skip
            # Here we skip incomplete tail window
            break
        windows.append((x, y))
    return windows

class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, windows):
        self.windows = windows
    def __len__(self):
        return len(self.windows)
    def __getitem__(self, idx):
        x, y = self.windows[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
```

**HF Gemma tokenizer/model ile embedding alma**
```python
# English comments
from transformers import AutoTokenizer, Gemma3ForCausalLM
import torch

text = "the capital of united states and the capital of france"
tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
model = Gemma3ForCausalLM.from_pretrained("google/gemma-3-1b-it")
ids = tok.encode(text)
emb = model.model.embed_tokens(torch.tensor(ids))  # [seq_len, hidden]
```

**Plotly ile 3B serpiştirme (örnek)**
```python
# English comments
import numpy as np
import plotly.graph_objects as go
import plotly.offline

def plot3d(points: np.ndarray, labels, title: str, dims=(0,1,2), color="red"):
    fig = go.Figure(data=[
        go.Scatter3d(
            x=points[:, dims[0]],
            y=points[:, dims[1]],
            z=points[:, dims[2]],
            mode="markers+text",
            text=labels,
            marker=dict(size=6, color=color),
            hoverinfo="text",
        )
    ])
    fig.update_layout(title=title)
    plotly.offline.iplot(fig)
```

---

## 13) Sonraki adımlar
- Basit bir **Causal LM** bloğu (Embedding → birkaç self-attention/MLP → LM Head) yazıp bu DataLoader çıktılarıyla *gerçek* eğitim deneyin.
- Loss olarak `CrossEntropy` kullanın; optimizer olarak `AdamW` tercih edin.
- Kararlılık için `gradient clipping` ve öğrenme oranı planlayıcıları (warmup/cosine) ekleyin.

---

## Kapanış
Bu notebook, dil modellemenin kalbini oluşturan üç kavramı — **tokenization**, **context window/pencereleme**, **embedding** — küçük, çalışır örneklerle gösteriyor. Buradaki akışı takip ederek kısa sürede kendi mini veri hattınızı ve görselleştirme araçlarınızı kurabilir, ardından gerçek bir modelin embedding katmanını inceleyerek derinleşebilirsiniz.

