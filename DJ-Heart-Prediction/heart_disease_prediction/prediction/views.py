from django.shortcuts import render

def home_view(request):
    """
    Renders the home page of the website.
    """
    context = {
        "title": "Home Page",
        "welcome_message": "Welcome to our website!",
    }
    return render(request, "home.html", context)
