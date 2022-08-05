import torch
import numpy as np

if __name__ == "__main__":
  #　１、验证torch.expand
  # tgt_words = torch.tensor([[1,2,3],
  #               [4,5,6],
  #               [7,8,9]])
  # tgt_pad_mask = tgt_words.unsqueeze(1)
  # print(tgt_pad_mask)
  # tgt_pad_mask = tgt_pad_mask.expand(3, 3, 3)
  # print(tgt_pad_mask)

  # 2、验证torch.split()
  scores = torch.Tensor([[[1,2,3],
              [4,5,6]],
              [[1,1,1],
              [0,0,0]]])
  # torch.split(tensor,split_szie,dim），split_size有整数，也有列表，dim默认为0，
  # res = torch.split(scores, 2)
  # print(res)


  # mask=torch.Tensor([[False, True, True],
  #           [False, False, True],
  #           [False, False, False]])
  # res = scores.masked_fill(mask, -1e18)
  # print(res)
  # src = torch.arange(1, 11).reshape((2, 5))
  # index = torch.tensor([[0, 1, 2], [0, 1, 4]])
  # res = torch.zeros(3, 5, dtype=src.dtype).scatter_(1, index, src)
  # print(res)
  
  # 3、验证predictor.py
  # beam_size = 4
  # batch_size = 3
  log_probs = torch.FloatTensor([[0,1,2],
                [1,2,3],
                [2,3,4],
                [3,4,5],
                [4,5,6],
                [5,6,7],
                [6,7,8],
                [7,8,9],
                [8,9,10],
                [9,10,11],
                [10,11,12],
                [11,12,13]])
  # # print(log_probs)
  # # exit()
  # topk_log_probs = (torch.tensor([0.0] + [float("-inf")] * (beam_size - 1)).repeat(batch_size))
  # # tensor([0., -inf, -inf, -inf, -inf])
  # log_probs += topk_log_probs.view(-1).unsqueeze(1)
  # print(log_probs)

  # 4、
  # topk_beam_index=torch.tensor([[3, 0, 0, 1, 1],
  #       [0, 1, 3, 0, 0],
  #       [0, 0, 2, 0, 0],
  #       [2, 4, 0, 0, 0],
  #       [0, 0, 1, 1, 0],
  #       [2, 3, 3, 3, 0]])
  # beam_offset=torch.tensor([0, 5, 10, 15, 20, 25])
  # batch_index = (topk_beam_index + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))

  # print(batch_index)

  # 5、
  is_finished = torch.tensor([[False, False, False, False, False],
        [False, True, False, False, False],
        [False, True, False, False, False],
        [False, True, False, False, False],
        [False, True, False, True, True],
        [True, True, False, False, False]])
  finished_hyp = is_finished[4].nonzero().view(-1)
  print("is_finished[i].nonzero()={}".format(is_finished[4].nonzero()))
  print("finished_hyp={}".format(finished_hyp))





