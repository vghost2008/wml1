_Pragma("once")
#ifdef GOOGLE_CUDA
void do_nms_classes_wise(const float* bboxes, const int* classes,const float threshold,bool* keep_mask,int data_nr);
#else
#error "error"
#endif
