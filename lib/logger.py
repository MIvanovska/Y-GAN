import os
import time

class Logger():

    def __init__(self, opt):

        self.opt = opt
        # Log file.
        self.log_name = os.path.join(opt.outf, opt.name, opt.mode, 'loss_log.txt')
        now  = time.strftime("%c")
        title = f'================ {now} ================\n'
        info = f'dataset={opt.dataset}, abnormal class={opt.abnormal_class}, batch size={opt.batchsize}, nz={opt.nz}, niter={opt.niter}, nperm={opt.nperm}, w_adv={opt.w_adv}, w_rec={opt.w_rec}, w_sem={opt.w_sem}, w_res={opt.w_res}, w_perm={opt.w_perm}\n'
        self.write_to_log_file(text=title + info)

    def print_current_errors(self, epoch, errors):
        message = '   Loss: [%d/%d] ' % (epoch, self.opt.niter)
        for key, val in errors.items():
            message += '%s: %.3f ' % (key, val)
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    ##
    def write_to_log_file(self, text):
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % text)

    ##
    def print_current_performance(self, performance, best):

        message = ''
        for key, val in performance.items():
            message += '%s: %.4f ' % (key, val)
        message += 'max AUC: %.4f' % best

        print(message)
        self.write_to_log_file(text=message)

    def print_test_performance(self, performance):

        message = ''
        for key, val in performance.items():
            message += '%s: %.4f ' % (key, val)

        print(message)
        self.write_to_log_file(text=message)