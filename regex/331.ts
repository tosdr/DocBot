import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*Date)|(?=.*Effective)|(?=.*last modified)|(?=.*Effective\:)|(?=.*updated\:))((?=.*update)|(?=.*last))", "i"),
	caseID: 331
} as Regex;