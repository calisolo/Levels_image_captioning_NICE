from transformers import Trainer

class ScstTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        ## eval sample output to get reward
        gen_target = [] #생성 토큰
        gen_res = [] #디코드 문장
        gt_res = [] # 정답문장
        for batch in train_dataloader:

          model.eval()
          with torch.no_grad():
            output = model.generate(input_ids = batch['input_ids'], patch_images = batch['patch_images'],num_beams=5, no_repeat_ngram_size=3, max_length = 40,
                                      use_cache=False  )
            gen_sentence = tokenizer.batch_decode(output, skip_special_tokens=True)
            gen_target.append(output)
            gen_res.append(gen_sentence)
            for _ in range(4):
              output = model.generate(input_ids = batch['input_ids'], patch_images = batch['patch_images'],temperature = 2.0,
                                      do_sample =  True,max_length = 40,
                                    #num_beams=5, no_repeat_ngram_size=3, 
                                      use_cache=False )
              gen_sentence = tokenizer.batch_decode(output, skip_special_tokens=True)
              gen_target.append(output)
              gen_res.append(gen_sentence)
          caption = batch['gt']
          gt_index = batch['gt_index']
          gt_res = caption
        dict_gt,dict_gen = {},{}
        for index, id in enumerate(gt_index):
          for i in range(len(gen_res)):
            dict_gen[str(id)+'_'+ str(i%5)] = [gen_res[i][index]]
            dict_gt[str(id)+'_'+ str(i%5)] = [gt_res[index]]

        cider,batch_cider_scores = Cider().compute_score(dict_gt, dict_gen)
        CIDER_REWARD_WEIGHT =1
        batch_cider_scores
        scores = CIDER_REWARD_WEIGHT * batch_cider_scores

        #scst reward
        batch_size = len(gt_res)
        gen_res_size = len(gen_res)*batch_size
        seq_per_img = gen_res_size // batch_size

        sc_ = scores.reshape(batch_size, seq_per_img)
        baseline = (sc_.sum(1, keepdims=True) - sc_) / (sc_.shape[1] - 1)
        reward = scores.reshape(batch_size, seq_per_img)
        reward = reward - baseline
        reward = reward.reshape(gen_res_size)
        reward = torch.as_tensor(reward, device=device, dtype=torch.float64)
        
        # model.train to get net output
        # batch_size = len(gt_res)
        # gen_res_size = len(gen_res)*batch_size
        # seq_per_img = gen_res_size // batch_size

        model.train()
        sample_src_tokens = torch.repeat_interleave(
            batch['input_ids'], seq_per_img, dim=0
        )
        sample_patch_images = torch.repeat_interleave(
                batch['patch_images'], seq_per_img, dim=0
        )

        len_max = 0
        for i in gen_target:
          if i.size(1) >len_max:
            len_max = i.size(1)
        for j in range(batch_size):
          for i in range(seq_per_img ):
            padding = torch.ones(len_max - len(gen_target[i][j]), dtype = torch.int64)
            gen_padded = torch.cat((gen_target[i][j], padding))
            if i==0 and j==0:
              gen_prev_output_tokens = gen_padded.unsqueeze(0)
            else:
              gen_prev_output_tokens = torch.cat((gen_prev_output_tokens, gen_padded.unsqueeze(0)),dim=0)

        net_output = model(input_ids = sample_src_tokens, patch_images = sample_patch_images, decoder_input_ids = gen_prev_output_tokens)
        # compute custom loss 
        lprobs = model.get_normalized_probs(net_output, log_probs=True) 
        loss = -lprobs.gather(dim=-1, index=gen_prev_output_tokens.unsqueeze(-1)).squeeze() * reward.unsqueeze(-1)
        pad_mask= gen_prev_output_tokens.eq(1)
        loss.masked_fill_(pad_mask, 0.0)
        ntokens = (~pad_mask).sum()
        loss = loss.sum()
        return (loss, net_output) if return_outputs else loss


