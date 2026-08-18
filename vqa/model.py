#--------------------------------------------------------------------------------------------<VQA Model>-----------------||
import json
import torch
import torch.nn as nn

from typing import Optional
from transformers import AutoModel

from vqa import config as C

# Late fusion architecture
class VQAClass(nn.Module):

    '''
    - Late fusion VQA Architecture
    - Collects Visual & Textual Encoder features and fuses them together.
    - Classifier predicts the answer label, cross entropy against the consensus answer.
    - Interfaces with Huggingface Trainer via the dict returned from forward().
    '''

    def __init__(self,
                 nLabels: int,
                 fusionDim: int = C.FUSION_DIM,
                 dropout: float = C.DROPOUT,
                 pretrainedT: str = C.TEXT_MODEL,
                 pretrainedV: str = C.IMAGE_MODEL,
                 freezeEncoders: bool = False):
        super(VQAClass, self).__init__()

        #--------------------------------------------------------------------------------<BERT + ViT>---------------------||
        self.pretrained_text_name  = pretrainedT # Pretrained Textual encoder
        self.pretrained_image_name = pretrainedV # Pretrained Visual encoder
        self.text_encoder  = AutoModel.from_pretrained(self.pretrained_text_name)
        self.image_encoder = AutoModel.from_pretrained(self.pretrained_image_name)

        #--------------------------------------------------------------------------------<Optional freeze>----------------||
        # Baseline mode
        if freezeEncoders:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        #--------------------------------------------------------------------------------<Fuse features>------------------||
        self.fusion = nn.Sequential(   # Fusion layer
            nn.Linear(self.text_encoder.config.hidden_size + self.image_encoder.config.hidden_size, fusionDim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        #--------------------------------------------------------------------------------<Classifier>---------------------||
        self.num_labels = nLabels   #answer space
        self.classifier = nn.Linear(fusionDim, self.num_labels)
        self.criterion  = nn.CrossEntropyLoss(ignore_index=0)  #index 0 is <unk>, excluded from the loss

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None):

        #Text Encoder
        encodeText = self.text_encoder(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            return_dict    = True,
        )

        #Visual Encoder
        encodeImage = self.image_encoder(
            pixel_values = pixel_values,
            return_dict  = True,
        )

        #Concatenate features
        fusedOut = self.fusion(
            torch.cat(
                [
                    encodeText["pooler_output"],    # Text
                    encodeImage["pooler_output"],   # Img
                ],
                dim=1
            )
        )

        #Classification
        logits = self.classifier(fusedOut)
        out = {"logits": logits}

        if labels is not None:
            out["loss"] = self.criterion(logits, labels)

        return out


#--------------------------------------------------------------------------------------------<LoRA Targets>--------------||
def getTargetModules(model):
    modules = [
        "text_encoder.embeddings.word_embeddings",
        "text_encoder.embeddings.position_embeddings",
        "text_encoder.embeddings.token_type_embeddings",
    ]

    nText = model.text_encoder.config.num_hidden_layers
    for layer in range(nText):
        for proj in ["query", "key", "value"]:
            modules.append(f"text_encoder.encoder.layer.{layer}.attention.self.{proj}")

    nImage = model.image_encoder.config.num_hidden_layers
    for layer in range(nImage):
        for proj in ["query", "key", "value"]:
            modules.append(f"image_encoder.encoder.layer.{layer}.attention.attention.{proj}")

    modules += ["text_encoder.pooler.dense", "image_encoder.pooler.dense", "fusion.0"]

    #----------------------------------------------------------------------------------------<Keep only real ones>-------||
    '''
    ViT renames its attention path between transformers versions, so drop anything
    that is not actually present rather than letting peft throw a ValueError'''
    
    present = set(name for name, _ in model.named_modules())
    kept    = [m for m in modules if m in present]
    dropped = len(modules) - len(kept)
    if dropped:
        print(f"[LoRA] {dropped} target module(s) not present in this model, skipped")
    print(f"[LoRA] targeting {len(kept)} modules")
    return kept


FINAL_MODULES = ["classifier"]   #trained in full, saved with the adapter


#--------------------------------------------------------------------------------------------<Declaring VQA + Fusing>----||
def declareModels(nLabels, useLoRA=False, freezeEncoders=False,
                  textModel=C.TEXT_MODEL, imageModel=C.IMAGE_MODEL):
    from vqa.dataProc import buildProcessor

    fdProcessor = buildProcessor(textModel, imageModel)
    vqaModel    = VQAClass(nLabels=nLabels, pretrainedT=textModel, pretrainedV=imageModel,
                           freezeEncoders=freezeEncoders).to(C.DEVICE)

    if useLoRA:
        import peft
        LConfig = peft.LoraConfig(
            r              = C.LORA_R,
            lora_alpha     = C.LORA_ALPHA,
            lora_dropout   = C.LORA_DROPOUT,
            target_modules = getTargetModules(vqaModel),   #lora layers
            modules_to_save= FINAL_MODULES,                #Final layer trained in full
        )
        vqaModel = peft.get_peft_model(vqaModel, LConfig)
        vqaModel.print_trainable_parameters()
    else:
        trainable = sum(p.numel() for p in vqaModel.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in vqaModel.parameters())
        print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100 * trainable / total:.4f}")

    return fdProcessor, vqaModel


#--------------------------------------------------------------------------------------------<Save + Load>---------------||
def saveModel(model, outDir, nLabels, useLoRA=False, freezeEncoders=False):
    # LoRA  -> adapter only (~40 MB)
    # woLoRA-> state_dict + a json describing how to rebuild the class
    outDir.mkdir(parents=True, exist_ok=True)

    meta = {
        "nLabels"        : nLabels,
        "textModel"      : C.TEXT_MODEL,
        "imageModel"     : C.IMAGE_MODEL,
        "fusionDim"      : C.FUSION_DIM,
        "dropout"        : C.DROPOUT,
        "useLoRA"        : useLoRA,
        "freezeEncoders" : freezeEncoders,
    }
    with open(outDir / "modelConfig.json", "w") as file:
        json.dump(meta, file, indent=2)

    if useLoRA:
        model.save_pretrained(str(outDir))
        print(f"Adapter saved -> {outDir}")
    else:
        torch.save(model.state_dict(), outDir / "vqaModel.pt")
        print(f"State dict saved -> {outDir / 'vqaModel.pt'}")

    return outDir


def loadModel(outDir):
    #----------------------------------------------------------------------------------------<Rebuild then load>-------||
    with open(outDir / "modelConfig.json", "r") as file:
        meta = json.load(file)

    vqaModel = VQAClass(
        nLabels        = meta["nLabels"],
        fusionDim      = meta["fusionDim"],
        dropout        = meta["dropout"],
        pretrainedT    = meta["textModel"],
        pretrainedV    = meta["imageModel"],
        freezeEncoders = meta.get("freezeEncoders", False),
    ).to(C.DEVICE)

    if meta["useLoRA"]:
        from peft import PeftModel
        vqaModel = PeftModel.from_pretrained(vqaModel, str(outDir))
        vqaModel = vqaModel.to(C.DEVICE)
        print(f"LoRA adapter loaded from {outDir}")
    else:
        state = torch.load(outDir / "vqaModel.pt", map_location=C.DEVICE)
        vqaModel.load_state_dict(state)
        print(f"State dict loaded from {outDir}")

    vqaModel.eval()
    return vqaModel, meta
