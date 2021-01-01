import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*one)|(?=.*1))((?=.*more than))((?=.*not)|(?=.*prohibit))((?=.*account))", "i"),
	caseID: 324,
	name: "Service does not allow alternative accounts"
} as Regex;