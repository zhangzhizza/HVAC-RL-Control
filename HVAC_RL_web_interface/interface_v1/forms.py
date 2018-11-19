from django import forms

class UploadFileForm(forms.Form):
	title = forms.CharField(label='Group name', max_length=50)
	file = forms.FileField(label='File location')

class SelectActionForm(forms.Form):
	def __init__(self, *args,**kwargs):
		try:
			self._options = kwargs.pop('options')
			super(SelectActionForm,self).__init__(*args,**kwargs)
			self.fields['ACTIONS'].choices = self._options;
		except Exception as e:
			pass;

	OPTIONS = (
            ("a", "A"),
            ("b", "B"),
            )
	ACTIONS = forms.MultipleChoiceField(widget=forms.CheckboxSelectMultiple,
                                             choices=OPTIONS)