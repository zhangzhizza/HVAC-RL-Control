from django import forms

class UploadFileForm(forms.Form):
	title = forms.CharField(label='Group Name', max_length=50)
	file_idf = forms.FileField(label='IDF File')
	file_epw = forms.FileField(label='EPW File', required=False)
	file_sch = forms.FileField(label='Schedule File', required=False)

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

class SelectSchForm(forms.Form):
	def __init__(self, *args,**kwargs):
		try:
			self._options = kwargs.pop('options')
			super(SelectSchForm,self).__init__(*args,**kwargs)
			self.fields['SCHEDULE'].choices = self._options;
		except Exception as e:
			pass;

	OPTIONS = (
            ("a", "A"),
            ("b", "B"),
            )
	SCHEDULE = forms.ChoiceField(widget=forms.Select(), choices=OPTIONS)

class SelectWeaForm(forms.Form):
	def __init__(self, *args,**kwargs):
		try:
			self._options = kwargs.pop('options')
			super(SelectWeaForm,self).__init__(*args,**kwargs)
			self.fields['EPW'].choices = self._options;
		except Exception as e:
			pass;

	OPTIONS = (
            ("a", "A"),
            ("b", "B"),
            )
	EPW = forms.ChoiceField(widget=forms.Select(), choices=OPTIONS)