from django import forms


class InputForm(forms.Form):
     messages = forms.CharField(label='質問文',max_length=50,
     min_length=1,widget=forms.Textarea(attrs=
     {'id': 'messages','size':'100', 'placeholder':'ここにドラゴンボールに関して質問したい文章を入力してください\n'})
     )