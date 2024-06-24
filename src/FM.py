class FM_Layer(nn.Module):
    def __init__(self, args, config):
        super(FM_Layer, self).__init__()
        input_dim = (args.num_layers + 1) * args.dim * 2
        # input_dim = args.dim * 2
        self.linear = nn.Linear(input_dim, 1, bias=False)
        self.V = nn.Parameter(
            t.zeros(input_dim, input_dim), requires_grad=True)
        self.bias_u = nn.Parameter(
            t.zeros(config['n_users'], requires_grad=True))
        self.bias_i = nn.Parameter(
            t.zeros(config['n_items'], requires_grad=True))
        self.bias = nn.Parameter(t.ones(1, requires_grad=False)*3)

        init.xavier_uniform_(self.V.data,0.1)

    def fm_layer(self, user_em, item_em, uid, iid):
        # linear_part: batch * 1 * input_dim
        x = t.cat((user_em, item_em), -1).unsqueeze(1)
        linear_part = self.linear(x).squeeze()
        batch_size = len(x)
        V = t.stack((self.V,) * batch_size)
        # batch * 1 * input_dim
        interaction_part_1 = t.bmm(x, V)  # (batch, 1, input_ dim)
        interaction_part_1 = t.pow(interaction_part_1, 2)
        interaction_part_2 = t.bmm(t.pow(x, 2), t.pow(V, 2))
        mlp_output = 0.5 * \
            t.sum((interaction_part_1 - interaction_part_2).squeeze(1), -1)
        rate = linear_part + mlp_output + \
            self.bias_u[uid] + self.bias_i[iid] + self.bias
        return rate

    def forward(self, user_em, item_em, uid, iid):
        return self.fm_layer(user_em, item_em, uid, iid).view(-1)