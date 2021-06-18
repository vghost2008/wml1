_Pragma("once")
/*
 * 是否特别处理边界上的anchors
 * Faster-RCNN原文认为不处理边界上的anchors会引起很大的收敛误差
 * 但在他们在Detectron/Detectron2的实现中默认都没有特别处理这种情况, 认为实际上没有什么影响
 */
//#define PROCESS_BOUNDARY_ANCHORS
#define MIN_SCORE_FOR_POS_BOX 0.0625
