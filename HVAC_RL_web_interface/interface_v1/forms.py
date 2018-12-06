from django import forms

class UploadFileForm(forms.Form):
	def __init__(self, *args, **kwargs):
		try:
			super(UploadFileForm,self).__init__(*args,**kwargs);
			self.fields['title'].widget.attrs['class'] = 'form-control form-control-sm';
			self.fields['file_idf'].widget.attrs['class'] = 'form-control-file';
			self.fields['file_epw'].widget.attrs['class'] = 'form-control-file';
			self.fields['file_sch'].widget.attrs['class'] = 'form-control-file';
		except Exception as e:
			pass;

	title = forms.CharField(label='Group Name', max_length=50)
	file_idf = forms.FileField(label='IDF File')
	file_epw = forms.FileField(label='EPW File', required=False)
	file_sch = forms.FileField(label='Schedule File', required=False)

class MinMaxLimitsForm(forms.Form):
	def __init__(self, *args, **kwargs):
		try:
			super(MinMaxLimitsForm,self).__init__(*args,**kwargs);
			self.fields['minm'].widget.attrs['class'] = 'form-control form-control-sm';
			self.fields['maxm'].widget.attrs['class'] = 'form-control form-control-sm';
			self.fields['minm'].widget.attrs['placeholder'] = 'Input the numbers, separated by comma';
			self.fields['maxm'].widget.attrs['placeholder'] = 'Input the numbers, separated by comma';
		except Exception as e:
			pass;

	minm = forms.CharField(label='Min', required=True)
	maxm = forms.CharField(label='Max', required=True)

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
			self.fields['SCHEDULE'].widget.attrs['class'] = 'form-control';
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
			self.fields['EPW'].widget.attrs['class'] = 'form-control';
		except Exception as e:
			pass;

	OPTIONS = (
            ("a", "A"),
            ("b", "B"),
            )
	EPW = forms.ChoiceField(widget=forms.Select(), choices=OPTIONS)