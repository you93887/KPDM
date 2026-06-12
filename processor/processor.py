import logging
from utils.metrics import Evaluator


def do_inference(model, test_img_loader, test_txt_loader):
    logger = logging.getLogger("RDE.test")
    logger.info("Enter inferencing")
    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())
