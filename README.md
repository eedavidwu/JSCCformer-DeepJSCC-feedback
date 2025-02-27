# Notes for Codes

## Running the Code

Run the following command:

```bash
python train.py
```

## Settings

- Select different **patch sizes** and their **positional encoding settings** for different data resolutions.
- Select different `args.tcn` for different **bandwidth ratios**.
- Select different feedback-block numbers for different settings.

## Block Selection

- For **CIFAR-10** and **CelebA**, this **Seq-to-Seq architecture** performs well with best performance, especially under lower complexity.
- Replacing the **learnable positional encoding** with **conditional positional encoding** benefits **resolution-adaptive design**.

## Some Tips

- For **large-resolution settings**, using an **alternative SWIN block** can speed up training.  
  _(Will update for convenient usage when available.)_

> **Note**: The code is currently uncleaned. I will clean it and make it more user-friendly when I have time.  

## Reference

```bibtex
@article{wu2024transformer,
  title={Transformer-aided wireless image transmission with channel feedback},
  author={Wu, Haotian and Shao, Yulin and Ozfatura, Emre and Mikolajczyk, Krystian and G{\"u}nd{\"u}z, Deniz},
  journal={IEEE Transactions on Wireless Communications},
  year={2024},
  publisher={IEEE}
}

@article{wu2025deep,
  title={Deep Joint Source and Channel Coding},
  author={Wu, Haotian and Bian, Chenghong and Shao, Yulin and G{"u}nd{"u}z, Deniz},
  journal={Foundations of Semantic Communication Networks},
  pages={61--110},
  year={2025},
  publisher={Wiley Online Library}
}
