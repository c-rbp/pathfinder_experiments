import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.nn import init
from torch.autograd import Function


class dummyhgru(Function):
    @staticmethod
    def forward(ctx, state_2nd_last, last_state, *args):
        ctx.save_for_backward(state_2nd_last, last_state)
        ctx.args = args
        return last_state

    @staticmethod
    def backward(ctx, grad):
        neumann_g = neumann_v = None
        neumann_g_prev = grad.clone()
        neumann_v_prev = grad.clone()

        state_2nd_last, last_state = ctx.saved_tensors

        args = ctx.args
        truncate_iter = args[-1]
        exp_name = args[-2]
        i = args[-3]
        epoch = args[-4]

        normsv = []
        normsg = []
        normg = torch.norm(neumann_g_prev)
        normsg.append(normg.data.item())
        normsv.append(normg.data.item())
        for ii in range(truncate_iter):
            neumann_v = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=neumann_v_prev,
                                            retain_graph=True, allow_unused=True)
            normv = torch.norm(neumann_v[0])
            neumann_g = neumann_g_prev + neumann_v[0]
            normg = torch.norm(neumann_g)

            if normg > 1 or normv > normsv[-1] or normv < 1e-9:
                normsg.append(normg.data.item())
                normsv.append(normv.data.item())
                neumann_g = neumann_g_prev
                break

            neumann_v_prev = neumann_v
            neumann_g_prev = neumann_g

            normsv.append(normv.data.item())
            normsg.append(normg.data.item())

        return (None, neumann_g, None, None, None, None)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        i_t = torch.sigmoid(self.Wxi(x) + self.Whi(h))  # + c * self.Wci)
        f_t = torch.sigmoid(self.Wxf(x) + self.Whf(h))  # + c * self.Wcf)
        c_t = f_t * c + i_t * torch.tanh(self.Wxc(x) + self.Whc(h))
        o_t = torch.sigmoid(self.Wxo(x) + self.Who(h))  # + cc * self.Wco)
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class ConvLSTM(nn.Module):

    def __init__(self, timesteps=8, filt_size=15, num_iter=50, exp_name='exp1', jacobian_penalty=False, grad_method='bptt'):
        # super().__init__()
        super(ConvLSTM, self).__init__()
        self.timesteps = timesteps
        self.num_iter = num_iter
        self.exp_name = exp_name
        self.jacobian_penalty = jacobian_penalty
        self.grad_method = grad_method
        
        self.conv0 = nn.Conv2d(1, 25, kernel_size=7, padding=3)
        part1 = np.load('utils/gabor_serre.npy')
        self.conv0.weight.data = torch.FloatTensor(part1)
        
        self.unit1 = ConvLSTMCell(25, 25, filt_size)
        print("Training with filter size:", filt_size, "x", filt_size)
        
        self.bn = nn.BatchNorm2d(25, eps=1e-03, track_running_stats=False)
        self.conv6 = nn.Conv2d(25, 2, kernel_size=1)
        init.xavier_normal_(self.conv6.weight)
        init.constant_(self.conv6.bias, torch.log(torch.tensor((1 - 0.01) / 0.01)))

    def forward(self, x, epoch, itr, target, criterion, testmode=False):

        x = self.conv0(x)
        x = torch.pow(x, 2)
        internal_h = torch.zeros_like(x, requires_grad=False)
        internal_c = torch.zeros_like(x, requires_grad=False)

        states = []
        if self.grad_method == 'rbp':
            with torch.no_grad():
                for i in range(self.timesteps - 1):
                    if testmode: states.append(internal_h)
                    (internal_h, internal_c) = self.unit1(x, internal_h, internal_c)
            if testmode: states.append(internal_h)
            state_2nd_last = internal_h.detach().requires_grad_()
            state_2nd_last_c = internal_c.detach().requires_grad_()
            i += 1
            (last_state, internal_c) = self.unit1(x, state_2nd_last, state_2nd_last_c)
            internal_h = dummyhgru.apply(state_2nd_last, last_state, epoch, itr, self.exp_name, self.num_iter)
            if testmode: states.append(internal_h)

        elif self.grad_method == 'bptt':
            for i in range(self.timesteps):
                (internal_h, internal_c) = self.unit1(x, internal_h, internal_c)
                if i == self.timesteps - 2:
                    state_2nd_last = internal_h
                    state_2nd_last_c = internal_c
                elif i == self.timesteps - 1:
                    last_state = internal_h

        output = self.bn(internal_h)
        output = self.conv6(output)
        loss = criterion(output, target)

        pen_type = 'l1'
        jv_penalty = torch.tensor([1]).float().cuda()
        mu = 0.90
        double_neg = False
        if self.training:   #and self.jacobian_penalty
            if pen_type == 'l1':
                norm_1_vect = torch.ones_like(last_state)
                norm_1_vect.requires_grad = False
                jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[norm_1_vect], retain_graph=True,
                                              create_graph=self.jacobian_penalty, allow_unused=True)[0]
                jv_penalty = (jv_prod - mu).clamp(0) ** 2

                jv_prod = torch.autograd.grad(internal_c, state_2nd_last_c, grad_outputs=[norm_1_vect], retain_graph=True,create_graph=self.jacobian_penalty, allow_unused=True)[0]
                jv_penalty = jv_penalty + (jv_prod - mu).clamp(0) ** 2

                if double_neg is True:
                    neg_norm_1_vect = -1 * norm_1_vect.clone()
                    jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[neg_norm_1_vect], retain_graph=True,
                                                  create_graph=True, allow_unused=True)[0]
                    jv_penalty2 = (jv_prod - mu).clamp(0) ** 2
                    jv_penalty = jv_penalty + jv_penalty2

        if testmode: return output, states, loss
        return output, jv_penalty, loss


if __name__ == '__main__':
    # gradient check
    # convlstm = ConvLSTM(input_channels=512, hidden_channels=[127, 32], kernel_size=3, step=5,
    #                     effective_step=[4]).cuda()
    model = ConvLSTM(timesteps=20, filt_size=15, num_iter=15, exp_name="args.name")
    loss_fn = torch.nn.MSELoss()
    import pdb; pdb.set_trace()
    input = Variable(torch.randn(1, 512, 64, 32)).cuda()
    target = Variable(torch.randn(1, 32, 64, 32)).double().cuda()

    output = convlstm(input)
    output = output[0][0].double()
    res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    print(res)
