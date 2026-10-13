"""Figure 4 fixed-weight LAPO training on cn_k12 (Qwen2.5-7B)."""


def main() -> None:
    from Train_CNK12_3B_Fixed import main as run_training

    run_training(
        run_model_size_label="7B",
        run_model_id="Qwen/Qwen2.5-7B-Instruct",
        run_sft_model_id="xw1234gan/SFT_Qwen2.5-7B-Instruct_cnk12",
        run_batch_gen_size=16,
    )


if __name__ == "__main__":
    main()
