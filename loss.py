"""
This file contains the implementation of DINO loss, which is used to train the DINO model.
Includes: 
- DINO Loss
"""
import torch
import torch.nn.functional as F
import torch.nn as nn

class DINOLoss(nn.Module):
    """
    DINO Loss with optimal centering (v2+)

    student_temp : sharper distribution for student.
    teacher_temp : smoother distribution for teacher ( Target).
    center_momentum : for updating center vector.
    use_centering : use centering or not. enabled for v2+, disabled for v1.

    """

    def __init__(self,
                 out_dim          = 128,
                 student_temp     = 0.1,
                 teacher_temp     = 0.04,
                 center_momentum  = 0.9,
                 use_centering    = False,
    ):

        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.use_centering = use_centering

        if use_centering:
            self.register_buffer(name ="center", tensor = torch.zeros(1, out_dim))

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        EMA update of center vector to prevent collapse
        """
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center  = (self.center_momentum * self.center
                        + (1 - self.center_momentum) * batch_center)

    def forward(self, student_output, teacher_outputs):
        """
        Student_outputs : list of (B, out_dim) for all views.
        teacher_outputs : list of (B, out_dim) for global views.
        """

        # Teacher: center + temperture
        teacher_out = torch.stack(teacher_outputs) # (n_global, B, D)
        if self.use_centering:
            teacher_out = teacher_out - self.center
            
        teacher_probs = F.softmax(teacher_out / self.teacher_temp, dim=-1).detach()

        # Student: temp
        student_out = torch.stack(student_output) # (n_views, B, D)
        student_log_probs = F.log_softmax(student_out / self.student_temp, dim=-1)

        # Cross-entropy loss for each student view against all teacher views
        total_loss = 0.0
        n_pairs = 0
        n_teacher = len(teacher_outputs)

        for t_idx in range(n_teacher):
            for s_idx in range(len(student_output)):

                if s_idx == t_idx:
                    continue # skip some-view pairs

                loss = -(teacher_probs[t_idx] * student_log_probs[s_idx]).sum(dim=-1)
                total_loss += loss.mean()
                n_pairs += 1
        
        loss = total_loss / n_pairs
     
        # Update center after computing loss
        if self.use_centering:
            self.update_center(torch.stack(teacher_outputs).mean(dim=0))
            
        return loss