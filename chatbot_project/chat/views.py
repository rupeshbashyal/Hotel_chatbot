import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_http_methods

from bot import get_response


@ensure_csrf_cookie
def home(request):
    return render(request, "index.html")


@require_http_methods(["POST"])
def get_bot_response(request):
    data = json.loads(request.body)
    message = data.get("message", "")
    response = get_response(message)
    return JsonResponse({"response": response})



