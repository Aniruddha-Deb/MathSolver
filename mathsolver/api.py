from flask import Blueprint, request
import base64
import io
from mathsolver.ai import network
from PIL import Image, ImageOps

bp = Blueprint('api', __name__, url_prefix='/api')

@bp.route('/get_answer', methods=['POST'])
def get_answer():
    print(request.form)
    img_bytes = base64.b64decode(request.form.get('img'))
    img_buf = io.BytesIO(img_bytes)
    img = ImageOps.grayscale(Image.open(img_buf))
    return str(network.solve(img))