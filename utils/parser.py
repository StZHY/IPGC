import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="KGIN")

    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="last-fm", help="Choose a dataset:[last-fm, movie-1m]")
    parser.add_argument(
        "--data_path", nargs="?", default="../data/", help="Input data path."
    )

    # ===== train ===== #
    parser.add_argument('--epoch', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--sim_regularity', type=float, default=1e-5, help='regularization weight for latent factor')
    parser.add_argument("--inverse_r", type=bool, default=True, help="consider inverse relation or not")
    parser.add_argument("--node_dropout", type=bool, default=True, help="consider node dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=0.5, help="ratio of node dropout")
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--channel", type=int, default=64, help="hidden channels for model")
    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[5, 10, 20, 50, 100]', help='Output sizes of every layer')
    parser.add_argument('--test_flag', nargs='?', default='part', help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument("--n_factors", type=int, default=4, help="number of latent factor for user favour")
    parser.add_argument("--ind", type=str, default='distance', help="Independence modeling: mi, distance, cosine")

    # ===== relation context ===== #
    parser.add_argument('--context_hops', type=int, default=3, help='number of context hops')

    # ===== save model ===== #
    parser.add_argument("--save", type=bool, default=False, help="save model or not")
    parser.add_argument("--out_dir", type=str, default="./weights/", help="output directory for model")
    parser.add_argument("--training_log", type=bool, default=True, help="log")

    # ===== critiquing model ===== #
    parser.add_argument('--cri_epoch', type=int, default=10, help='number of epochs')
    parser.add_argument('--cri_batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--cri_lr', type=float, default=0.005, help='critique_learning rate')
    parser.add_argument('--item_rank_num', type=int, default=5, help='number of top_k items')
    parser.add_argument('--cri_key_rank_num', type=int, default=5, help='number of need rank keyphrase')
    parser.add_argument('--reg_lambda', type=float, default=1e-2, help='Omega regularization')
    parser.add_argument("--using_omega", type=bool, default=True, help="using omega or not")
    parser.add_argument('--count_nhop', type=int, default=1, help='considering how many hops of cri_keyphrase')
    parser.add_argument('--walk_steps', type=int, default=1, help='random walk steps')
    parser.add_argument('--rand_item_num', type=int, default=5, help='random chosen item num')
    parser.add_argument('--r_probability', type=float, default=1.0, help='r probability')

    return parser.parse_args()
