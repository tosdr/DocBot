import { Regex } from '../models';

module.exports = {
    	expression: new RegExp("^(((?=.*class action)|(?=.*class-action))|((?=.*individual capacity)|(?=.*individual basis))|((?=.*not)|(?=.*waive))((?=.*class, )))", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 117,
	name: "You waive your right to a class action."
} as Regex;