
import dynet as dy

class RecursiveNN(dy.Saveable):

    def __init__(self, node_rep_size, nonterm_emb_size, model):
        self.U_adj = model.add_parameters((node_rep_size, node_rep_size*2+1))
        self.u_adj = model.add_parameters(node_rep_size)
        self.U_pro = model.add_parameters((node_rep_size, node_rep_size+nonterm_emb_size))
        self.u_pro = model.add_parameters(node_rep_size)
        self.dropout_rate = 0.0
        self.dropout_enabled = False

    def set_dropout(self, dropout_rate):
        self.dropout_rate = dropout_rate

    def disable_dropout(self):
        self.dropout_enabled = False

    def enable_dropout(self):
        self.dropout_enabled = True

    def compose(self, l_c, l_h, r_c, r_h, arrow_is_to_the_left):
        if arrow_is_to_the_left:
            arrow_bit  = dy.inputVector([1])
        else:
            arrow_bit  = dy.inputVector([-1])
        input_rep = dy.concatenate([l_h, r_h, arrow_bit])
        U_adj = dy.parameter(self.U_adj)
        u_adj = dy.parameter(self.u_adj)
        new_h = U_adj*input_rep+u_adj
        new_c = None
        return new_c, new_h

    def promote(self, l_c, l_h, nonterm_emb):
        U_pro = dy.parameter(self.U_pro)
        u_pro = dy.parameter(self.u_pro)
        input_rep = dy.concatenate([l_h, nonterm_emb])
        new_h = dy.tanh(U_pro*input_rep+u_pro)
        new_c = None
        return new_c, new_h

    def get_components(self):
        return self.U_adj, self.u_adj, self.U_pro, self.u_pro

    def restore_components(self, components):
        self.U_adj, self.u_adj, self.U_pro, self.u_pro = components


class TreeLSTM(dy.Saveable):

    def __init__(self, node_rep_size, nonterm_emb_size, model):
        self.U_i_1     = model.add_parameters((node_rep_size, node_rep_size))
        self.U_i_2     = model.add_parameters((node_rep_size, node_rep_size))
        self.b_i       = model.add_parameters(node_rep_size)
        self.U_f_11    = model.add_parameters((node_rep_size, node_rep_size))
        self.U_f_12    = model.add_parameters((node_rep_size, node_rep_size))
        self.U_f_21    = model.add_parameters((node_rep_size, node_rep_size))
        self.U_f_22    = model.add_parameters((node_rep_size, node_rep_size))
        self.b_f       = model.add_parameters(node_rep_size)
        self.U_o_1     = model.add_parameters((node_rep_size, node_rep_size))
        self.U_o_2     = model.add_parameters((node_rep_size, node_rep_size))
        self.b_o       = model.add_parameters(node_rep_size)
        self.U_u_1     = model.add_parameters((node_rep_size, node_rep_size))
        self.U_u_2     = model.add_parameters((node_rep_size, node_rep_size))
        self.b_u       = model.add_parameters(node_rep_size)
        self.W_i       = model.add_parameters((node_rep_size, node_rep_size))
        self.d_i       = model.add_parameters(node_rep_size)
        self.W_f       = model.add_parameters((node_rep_size, node_rep_size))
        self.d_f       = model.add_parameters(node_rep_size)
        self.W_o       = model.add_parameters((node_rep_size, node_rep_size))
        self.d_o       = model.add_parameters(node_rep_size)
        self.W_u       = model.add_parameters((node_rep_size, node_rep_size))
        self.d_u       = model.add_parameters(node_rep_size)
        self.b_i_left  = model.add_parameters(node_rep_size)
        self.b_i_right = model.add_parameters(node_rep_size)
        self.b_f_left  = model.add_parameters(node_rep_size)
        self.b_f_right = model.add_parameters(node_rep_size)
        self.b_o_left  = model.add_parameters(node_rep_size)
        self.b_o_right = model.add_parameters(node_rep_size)
        self.b_u_left  = model.add_parameters(node_rep_size)
        self.b_u_right = model.add_parameters(node_rep_size)
        self.W_i_nt    = model.add_parameters((node_rep_size, nonterm_emb_size))
        self.W_f_nt    = model.add_parameters((node_rep_size, nonterm_emb_size))
        self.W_o_nt    = model.add_parameters((node_rep_size, nonterm_emb_size))
        self.W_u_nt    = model.add_parameters((node_rep_size, nonterm_emb_size))
        self.dropout_rate = 0.0
        self.dropout_enabled = False

    def set_dropout(self, dropout_rate):
        self.dropout_rate = dropout_rate

    def disable_dropout(self):
        self.dropout_enabled = False

    def enable_dropout(self):
        self.dropout_enabled = True


    def compose(self, c1, h1, c2, h2, arrow_is_to_the_left):
        U_i_1     = dy.parameter(self.U_i_1     )
        U_i_2     = dy.parameter(self.U_i_2     )
        b_i       = dy.parameter(self.b_i       )
        U_f_11    = dy.parameter(self.U_f_11    )
        U_f_12    = dy.parameter(self.U_f_12    )
        U_f_21    = dy.parameter(self.U_f_21    )
        U_f_22    = dy.parameter(self.U_f_22    )
        b_f       = dy.parameter(self.b_f       )
        U_o_1     = dy.parameter(self.U_o_1     )
        U_o_2     = dy.parameter(self.U_o_2     )
        b_o       = dy.parameter(self.b_o       )
        U_u_1     = dy.parameter(self.U_u_1     )
        U_u_2     = dy.parameter(self.U_u_2     )
        b_u       = dy.parameter(self.b_u       )
        b_u_left  = dy.parameter(self.b_u_left  )
        b_u_right = dy.parameter(self.b_u_right )
        b_i_left  = dy.parameter(self.b_i_left  )
        b_i_right = dy.parameter(self.b_i_right )
        b_f_left  = dy.parameter(self.b_f_left  )
        b_f_right = dy.parameter(self.b_f_right )
        b_o_left  = dy.parameter(self.b_o_left  )
        b_o_right = dy.parameter(self.b_o_right )
        if arrow_is_to_the_left:
            b_i_x = b_i_left
            b_o_x = b_o_left
            b_f_x = b_f_left
            b_u_x = b_u_left
        else:
            b_i_x = b_i_right
            b_o_x = b_o_right
            b_f_x = b_f_right
            b_u_x = b_u_right
        i = dy.logistic(b_i_x + U_i_1*h1 + U_i_2*h2 + b_i)
        f1 = dy.logistic(b_f_x + U_f_11*h1 + U_f_12*h2 + b_f)
        f2 = dy.logistic(b_f_x + U_f_21*h1 + U_f_22*h2 + b_f)
        o = dy.logistic(b_o_x + U_o_1*h1 + U_o_2*h2 + b_o)
        u = dy.tanh(b_u_x + U_u_1*h1 + U_u_2*h2 + b_u)
        if self.dropout_enabled:
            u = dy.dropout(u, self.dropout_rate)
        c_new = dy.cmult(i, u) + dy.cmult(f1, c1) + dy.cmult(f2, c2)
        h_new = dy.cmult(o, dy.tanh(c_new))

        return c_new, h_new

    def promote(self, c, h, nonterm_emb):
        W_i       = dy.parameter(self.W_i       )
        d_i       = dy.parameter(self.d_i       )
        W_f       = dy.parameter(self.W_f       )
        d_f       = dy.parameter(self.d_f       )
        W_o       = dy.parameter(self.W_o       )
        d_o       = dy.parameter(self.d_o       )
        W_u       = dy.parameter(self.W_u       )
        d_u       = dy.parameter(self.d_u       )
        W_i_nt    = dy.parameter(self.W_i_nt    )
        W_f_nt    = dy.parameter(self.W_f_nt    )
        W_o_nt    = dy.parameter(self.W_o_nt    )
        W_u_nt    = dy.parameter(self.W_u_nt    )
        i = dy.logistic(W_i_nt*nonterm_emb + W_i*h + d_i)
        f = dy.logistic(W_f_nt*nonterm_emb + W_f*h + d_f)
        o = dy.logistic(W_o_nt*nonterm_emb + W_o*h + d_o)
        u = dy.tanh(W_u_nt*nonterm_emb + W_u*h + d_u)
        if self.dropout_enabled:
            u = dy.dropout(u, self.dropout_rate)
        c_new = dy.cmult(i, u) + dy.cmult(f, c)
        h_new = dy.cmult(o, dy.tanh(c_new))

        return c_new, h_new

    def get_components(self):
        return self.U_i_1     , \
               self.U_i_2     , \
               self.b_i       , \
               self.U_f_11    , \
               self.U_f_12    , \
               self.U_f_21    , \
               self.U_f_22    , \
               self.b_f       , \
               self.U_o_1     , \
               self.U_o_2     , \
               self.b_o       , \
               self.U_u_1     , \
               self.U_u_2     , \
               self.b_u       , \
               self.W_i       , \
               self.d_i       , \
               self.W_f       , \
               self.d_f       , \
               self.W_o       , \
               self.d_o       , \
               self.W_u       , \
               self.d_u       , \
               self.b_i_left  , \
               self.b_i_right , \
               self.b_f_left  , \
               self.b_f_right , \
               self.b_o_left  , \
               self.b_o_right , \
               self.b_u_left  , \
               self.b_u_right , \
               self.W_i_nt    , \
               self.W_f_nt    , \
               self.W_o_nt    , \
               self.W_u_nt

    def restore_components(self, components):
        self.U_i_1     , \
        self.U_i_2     , \
        self.b_i       , \
        self.U_f_11    , \
        self.U_f_12    , \
        self.U_f_21    , \
        self.U_f_22    , \
        self.b_f       , \
        self.U_o_1     , \
        self.U_o_2     , \
        self.b_o       , \
        self.U_u_1     , \
        self.U_u_2     , \
        self.b_u       , \
        self.W_i       , \
        self.d_i       , \
        self.W_f       , \
        self.d_f       , \
        self.W_o       , \
        self.d_o       , \
        self.W_u       , \
        self.d_u       , \
        self.b_i_left  , \
        self.b_i_right , \
        self.b_f_left  , \
        self.b_f_right , \
        self.b_o_left  , \
        self.b_o_right , \
        self.b_u_left  , \
        self.b_u_right , \
        self.W_i_nt    , \
        self.W_f_nt    , \
        self.W_o_nt    , \
        self.W_u_nt = components

class HeadOnly(dy.Saveable):

    def __init__(self, node_rep_size, nonterm_emb_size, model):
        self.dummy1 = model.add_parameters(node_rep_size)
        self.dummy2 = model.add_parameters(node_rep_size)
        self.dropout_rate = 0.0
        self.dropout_enabled = False

    def set_dropout(self, dropout_rate):
        self.dropout_rate = dropout_rate

    def disable_dropout(self):
        self.dropout_enabled = False

    def enable_dropout(self):
        self.dropout_enabled = True

    def compose(self, l_c, l_h, r_c, r_h, arrow_is_to_the_left):
        dy.parameter(self.dummy1)
        dy.parameter(self.dummy2)
        return l_c, l_h

    def promote(self, c, h, nonterm_emb):
        return c, h

    def get_components(self):
        return self.dummy1, self.dummy2

    def restore_components(self, components):
        self.dummy1, self.dummy2 = components

class LeSTM(dy.Saveable):

    def __init__(self, node_rep_size, nonterm_emb_size, model):
        self.Wi_l_h = model.add_parameters((node_rep_size, node_rep_size))
        self.Wi_l_c = model.add_parameters((node_rep_size, node_rep_size))
        self.Wi_r_h = model.add_parameters((node_rep_size, node_rep_size))
        self.Wi_r_c = model.add_parameters((node_rep_size, node_rep_size))
        self.wi_bias_left  = model.add_parameters(node_rep_size)
        self.wi_bias_right = model.add_parameters(node_rep_size)

        self.Wf_l_h = model.add_parameters((node_rep_size, node_rep_size))
        self.Wf_l_c = model.add_parameters((node_rep_size, node_rep_size))
        self.Wf_r_h = model.add_parameters((node_rep_size, node_rep_size))
        self.Wf_r_c = model.add_parameters((node_rep_size, node_rep_size))
        self.wf_bias_left  = model.add_parameters(node_rep_size)
        self.wf_bias_right = model.add_parameters(node_rep_size)

        self.Wc_l = model.add_parameters((node_rep_size, node_rep_size))
        self.Wc_r = model.add_parameters((node_rep_size, node_rep_size))
        self.wc_bias_left  = model.add_parameters(node_rep_size)
        self.wc_bias_right = model.add_parameters(node_rep_size)

        self.Wo_l = model.add_parameters((node_rep_size, node_rep_size))
        self.Wo_r = model.add_parameters((node_rep_size, node_rep_size))
        self.Wo_c = model.add_parameters((node_rep_size, node_rep_size))
        self.wo_bias_left  = model.add_parameters(node_rep_size)
        self.wo_bias_right = model.add_parameters(node_rep_size)

        self.Wi_unary_c = model.add_parameters((node_rep_size, node_rep_size))
        self.Wi_unary_h = model.add_parameters((node_rep_size, node_rep_size))
        self.Wi_unary_n = model.add_parameters((node_rep_size, nonterm_emb_size))
        self.wi_unary_bias   = model.add_parameters(node_rep_size)

        self.Wf_unary_c = model.add_parameters((node_rep_size, node_rep_size))
        self.Wf_unary_h = model.add_parameters((node_rep_size, node_rep_size))
        self.Wf_unary_n = model.add_parameters((node_rep_size, nonterm_emb_size))
        self.wf_unary_bias   = model.add_parameters(node_rep_size)

        self.Wc_unary_n = model.add_parameters((node_rep_size, nonterm_emb_size))
        self.Wc_unary_h = model.add_parameters((node_rep_size, node_rep_size))
        self.wc_unary_bias = model.add_parameters(node_rep_size)

        self.Wo_unary = model.add_parameters((node_rep_size, node_rep_size))
        self.Wo_unary_n = model.add_parameters((node_rep_size, nonterm_emb_size))
        self.Wo_unary_c = model.add_parameters((node_rep_size, node_rep_size))
        self.wo_unary_bias = model.add_parameters(node_rep_size)

        self.dropout_rate = 0.0
        self.dropout_enabled = False

    def set_dropout(self, dropout_rate):
        self.dropout_rate = dropout_rate

    def disable_dropout(self):
        self.dropout_enabled = False

    def enable_dropout(self):
        self.dropout_enabled = True

    def compose(self, l_c, l_h, r_c, r_h, arrow_is_to_the_left):

        Wi_l_h = dy.parameter(self.Wi_l_h)
        Wi_l_c = dy.parameter(self.Wi_l_c)
        Wi_r_h = dy.parameter(self.Wi_r_h)
        Wi_r_c = dy.parameter(self.Wi_r_c)
        if arrow_is_to_the_left:
            wi = dy.parameter(self.wi_bias_left)
        else:
            wi = dy.parameter(self.wi_bias_right)
        Wf_l_h = dy.parameter(self.Wf_l_h)
        Wf_l_c = dy.parameter(self.Wf_l_c)
        Wf_r_h = dy.parameter(self.Wf_r_h)
        Wf_r_c = dy.parameter(self.Wf_r_c)
        if arrow_is_to_the_left:
            wf     = dy.parameter(self.wf_bias_left)
        else:
            wf     = dy.parameter(self.wf_bias_right)
        Wc_l   = dy.parameter(self.Wc_l  )
        Wc_r   = dy.parameter(self.Wc_r  )
        if arrow_is_to_the_left:
            wc     = dy.parameter(self.wc_bias_left)
        else:
            wc     = dy.parameter(self.wc_bias_right)
        Wo_l   = dy.parameter(self.Wo_l  )
        Wo_r   = dy.parameter(self.Wo_r  )
        Wo_c   = dy.parameter(self.Wo_c  )
        if arrow_is_to_the_left:
            wo     = dy.parameter(self.wo_bias_left)
        else:
            wo     = dy.parameter(self.wo_bias_right)

        i_l = dy.logistic(Wi_l_h*l_h + Wi_r_h*r_h + Wi_l_c*l_c + Wi_r_c*r_c+wi)
        i_r = dy.logistic(Wi_l_h*r_h + Wi_r_h*l_h + Wi_l_c*r_c + Wi_r_c*l_c+wi)

        f_l = dy.logistic(Wf_l_h*l_h + Wf_r_h*r_h + Wf_l_c*l_c + Wf_r_c*r_c+wf)
        f_r = dy.logistic(Wf_l_h*r_h + Wf_r_h*l_h + Wf_l_c*r_c + Wf_r_c*l_c+wf)

        new_u = dy.tanh(dy.cmult(Wc_l * l_h, i_l) + dy.cmult(Wc_r * r_h, i_r) + wc)
        if self.dropout_enabled:
            new_u = dy.dropout(new_u, self.dropout_rate)
        new_c = dy.cmult(f_l, l_c) + dy.cmult(f_r, r_c) + new_u

        o = dy.logistic(Wo_l * l_h + Wo_r * r_h + Wo_c * new_c + wo)

        new_h = dy.cmult(o, dy.tanh(new_c))

        return new_c, new_h

    def promote(self, c, h, nonterm_emb):
        Wi_unary_c = dy.parameter(self.Wi_unary_c)
        Wi_unary_h = dy.parameter(self.Wi_unary_h)
        wi_unary   = dy.parameter(self.wi_unary_bias)
        Wf_unary_c = dy.parameter(self.Wf_unary_c)
        Wf_unary_h = dy.parameter(self.Wf_unary_h)
        wf_unary   = dy.parameter(self.wf_unary_bias)
        Wc_unary   = dy.parameter(self.Wc_unary_h)
        wc_unary   = dy.parameter(self.wc_unary_bias)
        Wo_unary   = dy.parameter(self.Wo_unary  )
        Wo_unary_c = dy.parameter(self.Wo_unary_c)
        wo_unary   = dy.parameter(self.wo_unary_bias)
        Wi_unary_n = dy.parameter(self.Wi_unary_n)
        Wf_unary_n = dy.parameter(self.Wf_unary_n)
        Wc_unary_n = dy.parameter(self.Wc_unary_n)
        Wo_unary_n = dy.parameter(self.Wo_unary_n)

        i = dy.logistic(Wi_unary_c*c + Wi_unary_h*h + Wi_unary_n*nonterm_emb + wi_unary)
        f = dy.logistic(Wf_unary_c*c + Wf_unary_h*h + Wf_unary_n*nonterm_emb + wf_unary)

        new_u = dy.tanh(dy.cmult(i, Wc_unary*h + Wc_unary_n*nonterm_emb) + wc_unary)
        if self.dropout_enabled:
            new_u = dy.dropout(new_u, self.dropout_rate)
        new_c = dy.cmult(f, c) + new_u
        o = dy.logistic(Wo_unary*h + Wo_unary_c*c + Wo_unary_n*nonterm_emb + wo_unary)

        new_h = dy.cmult(o, dy.tanh(new_c))

        return new_c, new_h

    def get_components(self):
        return self.Wi_l_h     , \
        self.Wi_l_c     , \
        self.Wi_r_h     , \
        self.Wi_r_c     , \
        self.wi_bias_left  , \
        self.wi_bias_right , \
        self.Wf_l_h     , \
        self.Wf_l_c     , \
        self.Wf_r_h     , \
        self.Wf_r_c     , \
        self.wf_bias_left  , \
        self.wf_bias_right , \
        self.Wc_l       , \
        self.Wc_r       , \
        self.wc_bias_left  , \
        self.wc_bias_right , \
        self.Wo_l       , \
        self.Wo_r       , \
        self.Wo_c       , \
        self.wo_bias_left  , \
        self.wo_bias_right , \
        self.Wi_unary_c , \
        self.Wi_unary_h , \
        self.wi_unary_bias , \
        self.Wf_unary_c , \
        self.Wf_unary_h , \
        self.wf_unary_bias , \
        self.Wc_unary_h   , \
        self.wc_unary_bias , \
        self.Wo_unary   , \
        self.Wo_unary_c , \
        self.wo_unary_bias, \
        self.Wi_unary_n, \
        self.Wf_unary_n, \
        self.Wc_unary_n, \
        self.Wo_unary_n


    def restore_components(self, components):
        self.Wi_l_h     , \
        self.Wi_l_c     , \
        self.Wi_r_h     , \
        self.Wi_r_c     , \
        self.wi_bias_left  , \
        self.wi_bias_right , \
        self.Wf_l_h     , \
        self.Wf_l_c     , \
        self.Wf_r_h     , \
        self.Wf_r_c     , \
        self.wf_bias_left  , \
        self.wf_bias_right , \
        self.Wc_l       , \
        self.Wc_r       , \
        self.wc_bias_left  , \
        self.wc_bias_right , \
        self.Wo_l       , \
        self.Wo_r       , \
        self.Wo_c       , \
        self.wo_bias_left  , \
        self.wo_bias_right , \
        self.Wi_unary_c , \
        self.Wi_unary_h , \
        self.wi_unary_bias , \
        self.Wf_unary_c , \
        self.Wf_unary_h , \
        self.wf_unary_bias , \
        self.Wc_unary_h   , \
        self.wc_unary_bias , \
        self.Wo_unary   , \
        self.Wo_unary_c , \
        self.wo_unary_bias, \
        self.Wi_unary_n, \
        self.Wf_unary_n, \
        self.Wc_unary_n, \
        self.Wo_unary_n = components

