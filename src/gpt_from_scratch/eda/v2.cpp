#include <torch/torch.h>
#include <torch/nn/functional.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>

// Hyperparameters
const int batch_size = 64;
const int block_size = 256;
const int max_iters = 5000;
const int eval_interval = 500;
const float learning_rate = 3e-4;
const int eval_iters = 200;
const int n_embd = 384;
const int n_head = 6;
const int n_layer = 6;
const float dropout = 0.2;

// Vocabulary
std::vector<char> chars;
int vocab_size;
std::unordered_map<char, int> stoi;
std::unordered_map<int, char> itos;

// Data
std::vector<int64_t> train_data;
std::vector<int64_t> val_data;

// Model
torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

class HeadImpl : public torch::nn::Module {
public:
    HeadImpl(int head_size) {
        key = register_module("key", torch::nn::Linear(n_embd, head_size, false));
        query = register_module("query", torch::nn::Linear(n_embd, head_size, false));
        value = register_module("value", torch::nn::Linear(n_embd, head_size, false));
        tril = torch::tril(torch::ones(block_size, block_size));
        dropout = torch::nn::Dropout(dropout);
    }

    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor k = key(x);
        torch::Tensor q = query(x);
        torch::Tensor wei = torch::matmul(q, k.transpose(-2, -1)) * std::pow(n_embd, -0.5);
        wei.masked_fill_(tril[0][0].unsqueeze(0).unsqueeze(0) == 0, -std::numeric_limits<float>::infinity());
        wei = torch::softmax(wei, -1);
        wei = dropout(wei);
        torch::Tensor v = value(x);
        torch::Tensor out = torch::matmul(wei, v);
        return out;
    }

    torch::nn::Linear key;
    torch::nn::Linear query;
    torch::nn::Linear value;
    torch::Tensor tril;
    torch::nn::Dropout dropout;
};
TORCH_MODULE(Head);

class MultiHeadAttentionImpl : public torch::nn::Module {
public:
    MultiHeadAttentionImpl(int num_heads, int head_size) {
        for (int i = 0; i < num_heads; ++i) {
            heads.push_back(register_module("head" + std::to_string(i), Head(head_size)));
        }
        proj = register_module("proj", torch::nn::Linear(n_embd, n_embd));
        dropout = torch::nn::Dropout(dropout);
    }

    torch::Tensor forward(torch::Tensor x) {
        std::vector<torch::Tensor> outputs;
        for (auto& head : heads) {
            outputs.push_back(head(x));
        }
        torch::Tensor out = torch::cat(outputs, -1);
        out = dropout(proj(out));
        return out;
    }

    std::vector<Head> heads;
    torch::nn::Linear proj;
    torch::nn::Dropout dropout;
};
TORCH_MODULE(MultiHeadAttention);

class FeedForwardImpl : public torch::nn::Module {
public:
    FeedForwardImpl() {
        net = torch::nn::Sequential(
            torch::nn::Linear(n_embd, 4 * n_embd),
            torch::nn::ReLU(),
            torch::nn::Linear(4 * n_embd, n_embd),
            torch::nn::Dropout(dropout)
        );
    }

    torch::Tensor forward(torch::Tensor x) {
        return net->forward(x);
    }

    torch::nn::Sequential net;
};
TORCH_MODULE(FeedForward);

class BlockImpl : public torch::nn::Module {
public:
    BlockImpl() {
        sa = register_module("sa", MultiHeadAttention(n_head, n_embd / n_head));
        ffwd = register_module("ffwd", FeedForward(n_embd));
        ln1 = register_module("ln1", torch::nn::LayerNorm(n_embd));
        ln2 = register_module("ln2", torch::nn::LayerNorm(n_embd));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = x + sa->forward(ln1->forward(x));
        x = x + ffwd->forward(ln2->forward(x));
        return x;
    }

    MultiHeadAttention sa;
    FeedForward ffwd;
    torch::nn::LayerNorm ln1;
    torch::nn::LayerNorm ln2;
};
TORCH_MODULE(Block);

class BigramLanguageModelImpl : public torch::nn::Module {
public:
    BigramLanguageModelImpl() {
        token_embedding_table = register_module("token_embedding_table", torch::nn::Embedding(vocab_size, n_embd));
        position_embedding_table = register_module("position_embedding_table", torch::nn::Embedding(block_size, n_embd));
        for (int i = 0; i < n_layer; ++i) {
            blocks->push_back(register_module("block" + std::to_string(i), Block()));
        }
        ln_f = register_module("ln_f", torch::nn::LayerNorm(n_embd));
        lm_head = register_module("lm_head", torch::nn::Linear(n_embd, vocab_size));
    }

    torch::Tensor forward(torch::Tensor idx, torch::Tensor targets = nullptr) {
        int B = idx.size(0);
        int T = idx.size(1);

        torch::Tensor token_emb = token_embedding_table->forward(idx);
        torch::Tensor pos_emb = position_embedding_table->forward(torch::arange(T).to(device));
        torch::Tensor x = token_emb + pos_emb;
        for (auto& block : *blocks) {
            x = block->forward(x);
        }
        torch::Tensor logits = lm_head->forward(x);

        if (targets.defined()) {
            torch::Tensor loss = torch::nn::functional::cross_entropy(logits.view({-1, vocab_size}), targets.view({-1}));
            return logits, loss;
        }
        else {
            return logits, nullptr;
        }
    }

    torch::Tensor generate(torch::Tensor idx, int max_new_tokens) {
        for (int i = 0; i < max_new_tokens; ++i) {
            torch::Tensor idx_cond = idx.slice(1, -block_size, -1);
            torch::Tensor logits, _;
            std::tie(logits, _) = forward(idx_cond);
            logits = logits.select(1, -1);
            torch::Tensor probs = torch::softmax(logits, -1);
            torch::Tensor idx_next = torch::multinomial(probs, 1);
            idx = torch::cat({ idx, idx_next }, 1);
        }
        return idx;
    }

    torch::nn::Embedding token_embedding_table;
    torch::nn::Embedding position_embedding_table;
    torch::nn::Sequential blocks;
    torch::nn::LayerNorm ln_f;
    torch::nn::Linear lm_head;
};
TORCH_MODULE(BigramLanguageModel);

// Utility functions

std::vector<int64_t> read_data(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<int64_t> data;
    std::string line;
    while (std::getline(file, line)) {
        for (char c : line) {
            data.push_back(stoi[c]);
        }
    }
    return data;
}

void initialize_vocabulary() { // pain
    chars = { ' ', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/',
              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?',
              '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
              'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_',
              '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
              'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~' };

    vocab_size = chars.size();

    for (int i = 0; i < vocab_size; ++i) {
        stoi[chars[i]] = i;
        itos[i] = chars[i];
    }
}

int main() {
    initialize_vocabulary();

    train_data = read_data("train.txt");
    val_data = read_data("val.txt");

    torch::TensorOptions options(torch::kLong);
    torch::Tensor train_tensor = torch::from_blob(train_data.data(), { train_data.size() / block_size, block_size }, options).clone().to(device);
    torch::Tensor val_tensor = torch::from_blob(val_data.data(), { val_data.size() / block_size, block_size }, options).clone().to(device);

    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    BigramLanguageModel model;
    model->to(device);

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));

    int num_train_batches = train_tensor.size(0);
    int num_val_batches = val_tensor.size(0);

    torch::Tensor train_loss = torch::zeros({ max_iters });
    torch::Tensor val_loss = torch::zeros({ max_iters / eval_interval });

    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, num_train_batches - 1);

    for (int iter = 0; iter < max_iters; ++iter) {
        model->train();

        int idx = distribution(generator);
        torch::Tensor batch = train_tensor.slice(0, idx, idx + batch_size).clone().to(device);

        optimizer.zero_grad();
        torch::Tensor _, loss = model->forward(batch, batch);
        loss.backward();
        optimizer.step();

        train_loss[iter] = loss.item<float>();

        if (iter % eval_interval == 0) {
            model->eval();

            torch::Tensor val_total_loss = torch::zeros({ eval_iters });
            for (int i = 0; i < eval_iters; ++i) {
                idx = distribution(generator);
                torch::Tensor val_batch = val_tensor.slice(0, idx, idx + batch_size).clone().to(device);

                torch::Tensor _, val_batch_loss = model->forward(val_batch, val_batch);
                val_total_loss[i] = val_batch_loss.item<float>();
            }

            float avg_val_loss = val_total_loss.mean().item<float>();
            val_loss[iter / eval_interval] = avg_val_loss;

            std::cout << "Iteration: " << iter << " | Train Loss: " << loss.item<float>() << " | Val Loss: " << avg_val_loss << std::endl;
        }
    }

    torch::save(model, "model.pt");

    return 0;
}
